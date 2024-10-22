#include "structure/kernel/kernel/gemm.h"
#include "structure/context/context.h"
#include "structure/kernel/generator/gemm.h"
#include "structure/kernel/kernel/kernel.h"
#include "utils/float.h"
#include "utils/hash.h"
#include "utils/utils.h"
#include "worker/builder.h"
#include "worker/lower.h"
#include "worker/runner.h"

namespace cpu_transformers {
namespace kernel {

class GemmConstantBiasKernelGeneratorImpl
    : public GemmConstantBiasKernelGenerator {
public:
  GemmConstantBiasKernelGeneratorImpl(Meta &&lhs_meta, Meta &&rhs_meta,
                                      Meta &&output_meta, float64_t alpha,
                                      float64_t beta, bool transA, bool transB,
                                      Tensor &&bias);
  GemmConstantBiasKernelGeneratorImpl(
      const GemmConstantBiasKernelGeneratorImpl &generator) = delete;
  GemmConstantBiasKernelGeneratorImpl(
      GemmConstantBiasKernelGeneratorImpl &&generator) = default;
  virtual ~GemmConstantBiasKernelGeneratorImpl() = default;
  std::shared_ptr<DoubleInputsWithoutBufferKernel>
  YieldDoubleInputsWithoutBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<GemmConstantBiasKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetLhsMeta() const override;
  const Meta &GetRhsMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const GemmConstantBiasKernelGeneratorImpl &other) const;

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

  std::unordered_map<Key, std::shared_ptr<GemmConstantBiasKernel>, KeyHash,
                     KeyEqual>
      kernels_;
  const Meta lhs_meta_;
  const Meta rhs_meta_;
  const Meta output_meta_;
  const float64_t alpha_;
  const float64_t beta_;
  const bool transA_;
  const bool transB_;
  const Tensor bias_;
};

std::unique_ptr<GemmConstantBiasKernelGenerator>
GemmConstantBiasKernelGenerator::Make(Meta &&lhs_meta, Meta &&rhs_meta,
                                      Meta &&output_meta, float64_t alpha,
                                      float64_t beta, bool transA, bool transB,
                                      Tensor &&bias) {
  return std::make_unique<GemmConstantBiasKernelGeneratorImpl>(
      std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta), alpha,
      beta, transA, transB, std::move(bias));
}

GemmConstantBiasKernelGeneratorImpl::GemmConstantBiasKernelGeneratorImpl(
    Meta &&lhs_meta, Meta &&rhs_meta, Meta &&output_meta, float64_t alpha,
    float64_t beta, bool transA, bool transB, Tensor &&bias)
    : lhs_meta_(std::move(lhs_meta)), rhs_meta_(std::move(rhs_meta)),
      output_meta_(std::move(output_meta)), alpha_(alpha), beta_(beta),
      transA_(transA), transB_(transB), bias_(bias) {}

std::shared_ptr<DoubleInputsWithoutBufferKernel>
GemmConstantBiasKernelGeneratorImpl::YieldDoubleInputsWithoutBufferKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return Yield(lhs_layout, rhs_layout, output_layout);
}

std::shared_ptr<GemmConstantBiasKernel>
GemmConstantBiasKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  const Meta &lhs_meta = GetLhsMeta(), rhs_meta = GetRhsMeta(),
             output_meta = GetOutputMeta();
  auto it = kernels_.find({lhs_layout, rhs_layout, output_layout});
  if (it != kernels_.end()) {
    return it->second;
  }
  size_t min_time_cost = std::numeric_limits<size_t>::max();
  std::shared_ptr<GemmConstantBiasKernel> min_kernel = nullptr;
  for (llvm::SmallVector<Axis, 3> axes : GetAxesInAllOrders()) {
    Tensor bias = bias_;
    std::shared_ptr<GemmConstantBiasKernel> kernel =
        std::make_shared<GemmConstantBiasKernel>(
            alpha_, beta_, transA_, transB_, std::move(bias), std::move(axes));
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

const Meta &GemmConstantBiasKernelGeneratorImpl::GetLhsMeta() const {
  return lhs_meta_;
}

const Meta &GemmConstantBiasKernelGeneratorImpl::GetRhsMeta() const {
  return rhs_meta_;
}

const Meta &GemmConstantBiasKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string GemmConstantBiasKernelGeneratorImpl::GetKernelName() const {
  return GemmConstantBiasKernel::kKernelName;
}

size_t GemmConstantBiasKernelGeneratorImpl::GetHashCode() const {
  std::hash<float64_t> f64_hash;
  std::hash<bool> bool_hash;
  size_t hash = typeid(GemmConstantBiasKernelGeneratorImpl).hash_code();
  hash ^= lhs_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= output_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= f64_hash(alpha_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= f64_hash(beta_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= bool_hash(transA_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= bool_hash(transB_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= bias_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool GemmConstantBiasKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const GemmConstantBiasKernelGeneratorImpl *other_ptr =
          dynamic_cast<const GemmConstantBiasKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool GemmConstantBiasKernelGeneratorImpl::Equals(
    const GemmConstantBiasKernelGeneratorImpl &other) const {
  return GetLhsMeta() == other.GetLhsMeta() &&
         GetOutputMeta() == other.GetOutputMeta() && alpha_ == other.alpha_ &&
         beta_ == other.beta_ && transA_ == other.transA_ &&
         transB_ == other.transB_ && bias_ == other.bias_;
}

GemmConstantBiasKernelGeneratorImpl::Key::Key(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout)
    : lhs_layout(lhs_layout.begin(), lhs_layout.end()),
      rhs_layout(rhs_layout.begin(), rhs_layout.end()),
      output_layout(output_layout.begin(), output_layout.end()) {}

size_t GemmConstantBiasKernelGeneratorImpl::KeyHash::operator()(
    const GemmConstantBiasKernelGeneratorImpl::Key &key) const {
  size_t hash = typeid(GemmConstantBiasKernelGeneratorImpl::Key).hash_code();
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

bool GemmConstantBiasKernelGeneratorImpl::KeyEqual::operator()(
    const GemmConstantBiasKernelGeneratorImpl::Key &lhs,
    const GemmConstantBiasKernelGeneratorImpl::Key &rhs) const {
  return lhs.lhs_layout == rhs.lhs_layout && lhs.rhs_layout == rhs.rhs_layout &&
         lhs.output_layout == rhs.output_layout;
}

} // namespace kernel
} // namespace cpu_transformers
