#include "structure/kernel/generator/maxpool.h"
#include "structure/kernel/kernel/maxpool.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class MaxPoolKernelWithoutPaddingGeneratorImpl
    : public MaxPoolWithoutPaddingKernelGenerator {
public:
  MaxPoolKernelWithoutPaddingGeneratorImpl(Meta &&input_meta,
                                           Meta &&output_meta,
                                           std::vector<int64_t> kernel_shape,
                                           std::vector<int64_t> strides);
  MaxPoolKernelWithoutPaddingGeneratorImpl(
      const MaxPoolKernelWithoutPaddingGeneratorImpl &) = delete;
  MaxPoolKernelWithoutPaddingGeneratorImpl(
      MaxPoolKernelWithoutPaddingGeneratorImpl &&) = default;
  virtual ~MaxPoolKernelWithoutPaddingGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<MaxPoolWithoutPaddingKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const MaxPoolKernelWithoutPaddingGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const std::vector<int64_t> kernel_shape_;
  const std::vector<int64_t> strides_;
};

std::unique_ptr<MaxPoolWithoutPaddingKernelGenerator>
MaxPoolWithoutPaddingKernelGenerator::Make(Meta &&input_meta,
                                           Meta &&output_meta,
                                           std::vector<int64_t> kernel_shape,
                                           std::vector<int64_t> strides) {
  return std::make_unique<MaxPoolKernelWithoutPaddingGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), kernel_shape, strides);
}

MaxPoolKernelWithoutPaddingGeneratorImpl::
    MaxPoolKernelWithoutPaddingGeneratorImpl(Meta &&input_meta,
                                             Meta &&output_meta,
                                             std::vector<int64_t> kernel_shape,
                                             std::vector<int64_t> strides)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      kernel_shape_(kernel_shape), strides_(strides) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
MaxPoolKernelWithoutPaddingGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<MaxPoolWithoutPaddingKernel>
MaxPoolKernelWithoutPaddingGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  std::vector<int64_t> kernel_shape = kernel_shape_, strides = strides_;
  return std::make_shared<MaxPoolWithoutPaddingKernel>(std::move(kernel_shape),
                                                       std::move(strides));
}

const Meta &MaxPoolKernelWithoutPaddingGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &MaxPoolKernelWithoutPaddingGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string MaxPoolKernelWithoutPaddingGeneratorImpl::GetKernelName() const {
  return MaxPoolWithoutPaddingKernel::kKernelName;
}

size_t MaxPoolKernelWithoutPaddingGeneratorImpl::GetHashCode() const {
  std::hash<int64_t> i64_hasher;
  size_t hash = typeid(MaxPoolKernelWithoutPaddingGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  for (int64_t i : kernel_shape_) {
    hash ^= i64_hasher(i) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  for (int64_t i : strides_) {
    hash ^= i64_hasher(i) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  return hash;
}

bool MaxPoolKernelWithoutPaddingGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const MaxPoolKernelWithoutPaddingGeneratorImpl *p =
          dynamic_cast<const MaxPoolKernelWithoutPaddingGeneratorImpl *>(
              &other)) {
    return Equals(*p);
  }
  return false;
}

bool MaxPoolKernelWithoutPaddingGeneratorImpl::Equals(
    const MaxPoolKernelWithoutPaddingGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() &&
         kernel_shape_ == other.kernel_shape_ && strides_ == other.strides_;
}

} // namespace kernel
} // namespace fluidml
