#include "structure/kernel/generator/averagepool.h"
#include "structure/kernel/kernel/averagepool.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class AveragePoolKernelWithoutPaddingGeneratorImpl
    : public AveragePoolWithoutPaddingKernelGenerator {
public:
  AveragePoolKernelWithoutPaddingGeneratorImpl(
      Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&dilations,
      std::vector<int64_t> &&kernel_shape, std::vector<int64_t> &&strides);
  AveragePoolKernelWithoutPaddingGeneratorImpl(
      const AveragePoolKernelWithoutPaddingGeneratorImpl &) = delete;
  AveragePoolKernelWithoutPaddingGeneratorImpl(
      AveragePoolKernelWithoutPaddingGeneratorImpl &&) = default;
  virtual ~AveragePoolKernelWithoutPaddingGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<AveragePoolWithoutPaddingKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const AveragePoolKernelWithoutPaddingGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const std::vector<int64_t> dilations_;
  const std::vector<int64_t> kernel_shape_;
  const std::vector<int64_t> strides_;
};

std::unique_ptr<AveragePoolWithoutPaddingKernelGenerator>
AveragePoolWithoutPaddingKernelGenerator::Make(
    Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&dilations,
    std::vector<int64_t> &&kernel_shape, std::vector<int64_t> &&strides) {
  return std::make_unique<AveragePoolKernelWithoutPaddingGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(dilations),
      std::move(kernel_shape), std::move(strides));
}

AveragePoolKernelWithoutPaddingGeneratorImpl::
    AveragePoolKernelWithoutPaddingGeneratorImpl(
        Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&dilations,
        std::vector<int64_t> &&kernel_shape, std::vector<int64_t> &&strides)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      dilations_(std::move(dilations)), kernel_shape_(std::move(kernel_shape)),
      strides_(std::move(strides)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
AveragePoolKernelWithoutPaddingGeneratorImpl::
    YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                        llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<AveragePoolWithoutPaddingKernel>
AveragePoolKernelWithoutPaddingGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  std::vector<int64_t> dilations = dilations_, kernel_shape = kernel_shape_,
                       strides = strides_;
  return std::make_shared<AveragePoolWithoutPaddingKernel>(
      std::move(dilations), std::move(kernel_shape), std::move(strides));
}

const Meta &AveragePoolKernelWithoutPaddingGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &
AveragePoolKernelWithoutPaddingGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string
AveragePoolKernelWithoutPaddingGeneratorImpl::GetKernelName() const {
  return AveragePoolWithoutPaddingKernel::kKernelName;
}

size_t AveragePoolKernelWithoutPaddingGeneratorImpl::GetHashCode() const {
  std::hash<int64_t> hasher;
  size_t hash =
      typeid(AveragePoolKernelWithoutPaddingGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  for (int64_t i : kernel_shape_) {
    hash ^= hasher(i) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  for (int64_t i : strides_) {
    hash ^= hasher(i) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  return hash;
}

bool AveragePoolKernelWithoutPaddingGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const auto *p =
          dynamic_cast<const AveragePoolKernelWithoutPaddingGeneratorImpl *>(
              &other)) {
    return Equals(*p);
  }
  return false;
}

bool AveragePoolKernelWithoutPaddingGeneratorImpl::Equals(
    const AveragePoolKernelWithoutPaddingGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() &&
         kernel_shape_ == other.kernel_shape_ && strides_ == other.strides_;
}

} // namespace kernel
} // namespace fluidml
