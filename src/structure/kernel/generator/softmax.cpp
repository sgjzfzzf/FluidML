#include "structure/kernel/generator/softmax.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class SoftmaxKernelGeneratorImpl : public SoftmaxKernelGenerator {
public:
  SoftmaxKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                             int64_t axis);
  SoftmaxKernelGeneratorImpl(const SoftmaxKernelGeneratorImpl &generator) =
      delete;
  SoftmaxKernelGeneratorImpl(SoftmaxKernelGeneratorImpl &&generator) = default;
  virtual ~SoftmaxKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithBufferKernel> YieldSingleInputWithBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<SoftmaxKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const SoftmaxKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const int64_t axis_;
};

std::unique_ptr<SoftmaxKernelGenerator>
SoftmaxKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                             int64_t axis) {
  return std::make_unique<SoftmaxKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), axis);
}

SoftmaxKernelGeneratorImpl::SoftmaxKernelGeneratorImpl(Meta &&input_meta,
                                                       Meta &&output_meta,
                                                       int64_t axis)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      axis_(axis) {}

std::shared_ptr<SingleInputWithBufferKernel>
SoftmaxKernelGeneratorImpl::YieldSingleInputWithBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<SoftmaxKernel>
SoftmaxKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                  llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<SoftmaxKernel>(axis_);
}

const Meta &SoftmaxKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &SoftmaxKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string SoftmaxKernelGeneratorImpl::GetKernelName() const {
  return SoftmaxKernel::kKernelName;
}

size_t SoftmaxKernelGeneratorImpl::GetHashCode() const {
  std::hash<int64_t> i64_hash;
  size_t hash = typeid(SoftmaxKernelGeneratorImpl).hash_code();
  hash ^= input_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= output_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= i64_hash(axis_) + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool SoftmaxKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const SoftmaxKernelGeneratorImpl *other_ptr =
          dynamic_cast<const SoftmaxKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  }
  return false;
}

bool SoftmaxKernelGeneratorImpl::Equals(
    const SoftmaxKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ &&
         output_meta_ == other.output_meta_ && axis_ == other.axis_;
}

} // namespace kernel
} // namespace cpu_transformers
