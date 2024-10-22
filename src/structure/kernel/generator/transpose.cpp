#include "structure/kernel/generator/transpose.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class TransposeKernelGeneratorImpl : public TransposeKernelGenerator {
public:
  TransposeKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                               std::vector<int64_t> &&perms);
  TransposeKernelGeneratorImpl(const TransposeKernelGeneratorImpl &) = delete;
  TransposeKernelGeneratorImpl(TransposeKernelGeneratorImpl &&) = default;
  virtual ~TransposeKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<TransposeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const TransposeKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const std::vector<int64_t> perms_;
};

std::unique_ptr<TransposeKernelGenerator>
TransposeKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                               std::vector<int64_t> &&perms) {
  return std::make_unique<TransposeKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(perms));
}

TransposeKernelGeneratorImpl::TransposeKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&perms)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      perms_(std::move(perms)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
TransposeKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<TransposeKernel>
TransposeKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                    llvm::ArrayRef<size_t> output_layout) {
  std::vector<int64_t> perms = perms_;
  return std::make_shared<TransposeKernel>(std::move(perms));
}

const Meta &TransposeKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &TransposeKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string TransposeKernelGeneratorImpl::GetKernelName() const {
  return TransposeKernel::kKernelName;
}

size_t TransposeKernelGeneratorImpl::GetHashCode() const {
  std::hash<int64_t> i64_hash;
  size_t hash = typeid(TransposeKernelGeneratorImpl).hash_code();
  hash ^= input_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= output_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  for (int64_t perm : perms_) {
    hash ^= i64_hash(perm) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  return hash;
}

bool TransposeKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const TransposeKernelGeneratorImpl *other_ptr =
          dynamic_cast<const TransposeKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  }
  return false;
}

bool TransposeKernelGeneratorImpl::Equals(
    const TransposeKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ &&
         output_meta_ == other.output_meta_ && perms_ == other.perms_;
}

} // namespace kernel
} // namespace cpu_transformers