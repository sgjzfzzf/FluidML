#include "structure/kernel/generator/matmul.h"
#include "structure/kernel/kernel/matmul.h"
#include "utils/hash.h"
#include "utils/utils.h"
#include "worker/builder.h"
#include "worker/lower.h"
#include "worker/runner.h"

namespace cpu_transformers {
namespace kernel {

class MatMulKernelGeneratorImpl : public MatMulKernelGenerator {
public:
  MatMulKernelGeneratorImpl(Meta &&lhs_meta, Meta &&rhs_meta,
                            Meta &&output_meta);
  MatMulKernelGeneratorImpl(const MatMulKernelGeneratorImpl &generator) =
      delete;
  MatMulKernelGeneratorImpl(MatMulKernelGeneratorImpl &&generator) = default;
  virtual ~MatMulKernelGeneratorImpl() = default;
  std::shared_ptr<DoubleInputsWithoutBufferKernel>
  YieldDoubleInputsWithoutBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<MatMulKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetLhsMeta() const override;
  const Meta &GetRhsMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const MatMulKernelGeneratorImpl &other) const;

private:
  struct KeyHash;
  struct KeyEqual;

  class Key {
  public:
    Key(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout);
    Key(const Key &key) = default;
    Key(Key &&key) = default;
    ~Key() = default;
    friend struct KeyHash;
    friend struct KeyEqual;

  private:
    llvm::SmallVector<size_t> lhs_layout;
    llvm::SmallVector<size_t> rhs_layout;
    llvm::SmallVector<size_t> output_layout;
  };

  struct KeyHash {
    size_t operator()(const Key &key) const;
  };
  struct KeyEqual {
    bool operator()(const Key &lhs, const Key &rhs) const;
  };

  std::unordered_map<Key, std::shared_ptr<MatMulKernel>, KeyHash, KeyEqual>
      kernels_;
  const Meta lhs_meta_;
  const Meta rhs_meta_;
  const Meta output_meta_;
};

std::unique_ptr<MatMulKernelGenerator>
MatMulKernelGenerator::Make(Meta &&lhs_meta, Meta &&rhs_meta,
                            Meta &&output_meta) {
  return std::make_unique<MatMulKernelGeneratorImpl>(
      std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta));
};

MatMulKernelGeneratorImpl::MatMulKernelGeneratorImpl(Meta &&lhs_meta,
                                                     Meta &&rhs_meta,
                                                     Meta &&output_meta)
    : lhs_meta_(std::move(lhs_meta)), rhs_meta_(std::move(rhs_meta)),
      output_meta_(std::move(output_meta)) {}

std::shared_ptr<DoubleInputsWithoutBufferKernel>
MatMulKernelGeneratorImpl::YieldDoubleInputsWithoutBufferKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return Yield(lhs_layout, rhs_layout, output_layout);
}

std::shared_ptr<MatMulKernel>
MatMulKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> lhs_layout,
                                 llvm::ArrayRef<size_t> rhs_layout,
                                 llvm::ArrayRef<size_t> output_layout) {
  const Meta &lhs_meta = GetLhsMeta(), rhs_meta = GetRhsMeta(),
             output_meta = GetOutputMeta();
  // std::shared_ptr<MatMulKernel> kernel = std::make_shared<MatMulKernel>(
  //     llvm::SmallVector<Axis, 3>{Axis::i, Axis::j, Axis::k});
  // return kernel;
  auto it = kernels_.find({lhs_layout, rhs_layout, output_layout});
  if (it != kernels_.end()) {
    return it->second;
  }
  size_t min_time_cost = std::numeric_limits<size_t>::max();
  std::shared_ptr<MatMulKernel> min_kernel = nullptr;
  for (llvm::SmallVector<Axis, 3> axes : GetAxesInAllOrders()) {
    std::shared_ptr<MatMulKernel> kernel =
        std::make_shared<MatMulKernel>(std::move(axes));
    std::string kernel_name = kernel->GetKernelName();
    context::Context context;
    std::unique_ptr<worker::KernelBuilder> builder =
        context.MakeKernelBuilder(std::move(kernel_name));
    std::unique_ptr<worker::Lower> lower = context.MakeLower();
    std::unique_ptr<worker::Runner> runner = context.MakeRunner();
    builder->RunOnDoubleInputsWithoutBuffer(*kernel, lhs_meta, rhs_meta,
                                            output_meta);
    lower->Run();
    std::vector<uint8_t> lhs = utils::FillBuffer(lhs_meta),
                         rhs = utils::FillBuffer(rhs_meta),
                         output = utils::FillBuffer(output_meta);
    const size_t time_cost = runner->Run({
        {worker::KernelBuilder::kLhsKey, lhs.data()},
        {worker::KernelBuilder::kRhsKey, rhs.data()},
        {worker::KernelBuilder::kOutputKey, output.data()},
    });
    if (time_cost < min_time_cost) {
      min_time_cost = time_cost;
      min_kernel = kernel;
    }
  }
  kernels_.insert_or_assign({lhs_layout, rhs_layout, output_layout},
                            min_kernel);
#ifdef DEBUG
  assert(min_kernel != nullptr);
#endif
  return min_kernel;
}

const Meta &MatMulKernelGeneratorImpl::GetLhsMeta() const { return lhs_meta_; }

const Meta &MatMulKernelGeneratorImpl::GetRhsMeta() const { return rhs_meta_; }

const Meta &MatMulKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string MatMulKernelGeneratorImpl::GetKernelName() const {
  return MatMulKernel::kKernelName;
}

size_t MatMulKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(MatMulKernelGeneratorImpl).hash_code();
  hash ^= lhs_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= rhs_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= output_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool MatMulKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const MatMulKernelGeneratorImpl *other_ptr =
          dynamic_cast<const MatMulKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool MatMulKernelGeneratorImpl::Equals(
    const MatMulKernelGeneratorImpl &other) const {
  return lhs_meta_ == other.lhs_meta_ && rhs_meta_ == other.rhs_meta_ &&
         output_meta_ == other.output_meta_;
}

MatMulKernelGeneratorImpl::Key::Key(llvm::ArrayRef<size_t> lhs_layout,
                                    llvm::ArrayRef<size_t> rhs_layout,
                                    llvm::ArrayRef<size_t> output_layout)
    : lhs_layout(lhs_layout), rhs_layout(rhs_layout),
      output_layout(output_layout) {}

size_t MatMulKernelGeneratorImpl::KeyHash::operator()(
    const MatMulKernelGeneratorImpl::Key &key) const {
  size_t hash = 0;
  std::hash<size_t> hasher;
  for (size_t i = 0; i < key.lhs_layout.size(); ++i) {
    hash ^= hasher(key.lhs_layout[i]) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  for (size_t i = 0; i < key.rhs_layout.size(); ++i) {
    hash ^= hasher(key.rhs_layout[i]) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  for (size_t i = 0; i < key.output_layout.size(); ++i) {
    hash ^=
        hasher(key.output_layout[i]) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  return hash;
}

bool MatMulKernelGeneratorImpl::KeyEqual::operator()(
    const MatMulKernelGeneratorImpl::Key &lhs,
    const MatMulKernelGeneratorImpl::Key &rhs) const {
  return lhs.lhs_layout == rhs.lhs_layout && lhs.rhs_layout == rhs.rhs_layout &&
         lhs.output_layout == rhs.output_layout;
}

} // namespace kernel
} // namespace cpu_transformers
