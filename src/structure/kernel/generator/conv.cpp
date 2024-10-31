#include "structure/kernel/generator/conv.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class ConvWithoutPaddingKernelGeneratorImpl
    : public ConvWithoutPaddingKernelGenerator {
public:
  ConvWithoutPaddingKernelGeneratorImpl(Meta &&input_meta, Meta &&weights_meta,
                                        Meta &&output_meta,
                                        std::vector<int64_t> &&dilations,
                                        int64_t group,
                                        std::vector<int64_t> &&kernel_shape,
                                        std::vector<int64_t> &&strides,
                                        std::optional<Tensor> &&bias);
  ConvWithoutPaddingKernelGeneratorImpl(
      const ConvWithoutPaddingKernelGeneratorImpl &) = delete;
  ConvWithoutPaddingKernelGeneratorImpl(
      ConvWithoutPaddingKernelGeneratorImpl &&) = default;
  virtual ~ConvWithoutPaddingKernelGeneratorImpl() = default;
  std::shared_ptr<DoubleInputsWithoutBufferKernel>
  YieldDoubleInputsWithoutBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<ConvWithoutPaddingKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> weights_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetLhsMeta() const override;
  const Meta &GetRhsMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const ConvWithoutPaddingKernelGeneratorImpl &other) const;

private:
  const Meta lhs_meta_;
  const Meta rhs_meta_;
  const Meta output_meta_;
  const std::vector<int64_t> dilations_;
  const int64_t group_;
  const std::vector<int64_t> kernel_shape_;
  const std::vector<int64_t> strides_;
  const std::optional<Tensor> bias_;
};

class ConvWithPaddingKernelGeneratorImpl
    : public ConvWithPaddingKernelGenerator {
public:
  ConvWithPaddingKernelGeneratorImpl(
      Meta &&input_meta, Meta &&weights_meta, Meta &&output_meta,
      std::vector<int64_t> &&dilations, int64_t group,
      std::vector<int64_t> &&kernel_shape, std::vector<int64_t> &&pads,
      std::vector<int64_t> &&strides, std::optional<Tensor> &&bias);
  ConvWithPaddingKernelGeneratorImpl(
      const ConvWithPaddingKernelGeneratorImpl &) = delete;
  ConvWithPaddingKernelGeneratorImpl(ConvWithPaddingKernelGeneratorImpl &&) =
      default;
  virtual ~ConvWithPaddingKernelGeneratorImpl() = default;
  std::shared_ptr<DoubleInputsWithBufferKernel>
  YieldDoubleInputsWithBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<ConvWithPaddingKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> weights_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetLhsMeta() const override;
  const Meta &GetRhsMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const ConvWithPaddingKernelGeneratorImpl &other) const;

private:
  const Meta lhs_meta_;
  const Meta rhs_meta_;
  const Meta output_meta_;
  const std::vector<int64_t> dilations_;
  const int64_t group_;
  const std::vector<int64_t> kernel_shape_;
  const std::vector<int64_t> pads_;
  const std::vector<int64_t> strides_;
  const std::optional<Tensor> bias_;
};

std::unique_ptr<ConvWithoutPaddingKernelGenerator>
ConvWithoutPaddingKernelGenerator::Make(Meta &&input_meta, Meta &&weights_meta,
                                        Meta &&output_meta,
                                        std::vector<int64_t> &&dilations,
                                        int64_t group,
                                        std::vector<int64_t> &&kernel_shape,
                                        std::vector<int64_t> &&strides,
                                        std::optional<Tensor> &&bias) {
  return std::make_unique<ConvWithoutPaddingKernelGeneratorImpl>(
      std::move(input_meta), std::move(weights_meta), std::move(output_meta),
      std::move(dilations), group, std::move(kernel_shape), std::move(strides),
      std::move(bias));
}

std::unique_ptr<ConvWithPaddingKernelGenerator>
ConvWithPaddingKernelGenerator::Make(
    Meta &&input_meta, Meta &&weights_meta, Meta &&output_meta,
    std::vector<int64_t> &&dilations, int64_t group,
    std::vector<int64_t> &&kernel_shape, std::vector<int64_t> &&pads,
    std::vector<int64_t> &&strides, std::optional<Tensor> &&bias) {
  return std::make_unique<ConvWithPaddingKernelGeneratorImpl>(
      std::move(input_meta), std::move(weights_meta), std::move(output_meta),
      std::move(dilations), group, std::move(kernel_shape), std::move(pads),
      std::move(strides), std::move(bias));
}

ConvWithoutPaddingKernelGeneratorImpl::ConvWithoutPaddingKernelGeneratorImpl(
    Meta &&input_meta, Meta &&weights_meta, Meta &&output_meta,
    std::vector<int64_t> &&dilations, int64_t group,
    std::vector<int64_t> &&kernel_shape, std::vector<int64_t> &&strides,
    std::optional<Tensor> &&bias)
    : lhs_meta_(std::move(input_meta)), rhs_meta_(std::move(weights_meta)),
      output_meta_(std::move(output_meta)), dilations_(std::move(dilations)),
      group_(group), kernel_shape_(std::move(kernel_shape)),
      strides_(std::move(strides)), bias_(std::move(bias)) {}

std::shared_ptr<DoubleInputsWithoutBufferKernel>
ConvWithoutPaddingKernelGeneratorImpl::YieldDoubleInputsWithoutBufferKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return Yield(lhs_layout, rhs_layout, output_layout);
}

std::shared_ptr<ConvWithoutPaddingKernel>
ConvWithoutPaddingKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> weights_layout,
    llvm::ArrayRef<size_t> output_layout) {
  std::vector<int64_t> dilations = dilations_, kernel_shape = kernel_shape_,
                       strides = strides_;
  std::optional<Tensor> bias = bias_;
  return std::make_shared<ConvWithoutPaddingKernel>(
      std::move(dilations), group_, std::move(kernel_shape), std::move(strides),
      std::move(bias));
}

const Meta &ConvWithoutPaddingKernelGeneratorImpl::GetLhsMeta() const {
  return lhs_meta_;
}

const Meta &ConvWithoutPaddingKernelGeneratorImpl::GetRhsMeta() const {
  return rhs_meta_;
}

const Meta &ConvWithoutPaddingKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string ConvWithoutPaddingKernelGeneratorImpl::GetKernelName() const {
  return ConvWithoutPaddingKernel::kKernelName;
}

size_t ConvWithoutPaddingKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(ConvWithoutPaddingKernelGeneratorImpl).hash_code();
  hash ^= GetLhsMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetRhsMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool ConvWithoutPaddingKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const ConvWithoutPaddingKernelGeneratorImpl *other_ptr =
          dynamic_cast<const ConvWithoutPaddingKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool ConvWithoutPaddingKernelGeneratorImpl::Equals(
    const ConvWithoutPaddingKernelGeneratorImpl &other) const {
  return GetLhsMeta() == other.GetLhsMeta() &&
         GetRhsMeta() == other.GetRhsMeta() &&
         GetOutputMeta() == other.GetOutputMeta() &&
         dilations_ == other.dilations_ && group_ == other.group_ &&
         kernel_shape_ == other.kernel_shape_ && strides_ == other.strides_ &&
         bias_ == other.bias_;
}

ConvWithPaddingKernelGeneratorImpl::ConvWithPaddingKernelGeneratorImpl(
    Meta &&input_meta, Meta &&weights_meta, Meta &&output_meta,
    std::vector<int64_t> &&dilations, int64_t group,
    std::vector<int64_t> &&kernel_shape, std::vector<int64_t> &&pads,
    std::vector<int64_t> &&strides, std::optional<Tensor> &&bias)
    : lhs_meta_(std::move(input_meta)), rhs_meta_(std::move(weights_meta)),
      output_meta_(std::move(output_meta)), dilations_(std::move(dilations)),
      group_(group), kernel_shape_(std::move(kernel_shape)),
      pads_(std::move(pads)), strides_(std::move(strides)),
      bias_(std::move(bias)) {}

std::shared_ptr<DoubleInputsWithBufferKernel>
ConvWithPaddingKernelGeneratorImpl::YieldDoubleInputsWithBufferKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return Yield(lhs_layout, rhs_layout, output_layout);
}

std::shared_ptr<ConvWithPaddingKernel>
ConvWithPaddingKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> weights_layout,
    llvm::ArrayRef<size_t> output_layout) {
  std::vector<int64_t> dilations = dilations_, kernel_shape = kernel_shape_,
                       pads = pads_, strides = strides_;
  std::optional<Tensor> bias = bias_;
  return std::make_shared<ConvWithPaddingKernel>(
      std::move(dilations), group_, std::move(kernel_shape), std::move(pads),
      std::move(strides), std::move(bias));
}

const Meta &ConvWithPaddingKernelGeneratorImpl::GetLhsMeta() const {
  return lhs_meta_;
}

const Meta &ConvWithPaddingKernelGeneratorImpl::GetRhsMeta() const {
  return rhs_meta_;
}

const Meta &ConvWithPaddingKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string ConvWithPaddingKernelGeneratorImpl::GetKernelName() const {
  return ConvWithPaddingKernel::kKernelName;
}

size_t ConvWithPaddingKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(ConvWithPaddingKernelGeneratorImpl).hash_code();
  hash ^= GetLhsMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetRhsMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool ConvWithPaddingKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const ConvWithPaddingKernelGeneratorImpl *other_ptr =
          dynamic_cast<const ConvWithPaddingKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool ConvWithPaddingKernelGeneratorImpl::Equals(
    const ConvWithPaddingKernelGeneratorImpl &other) const {
  return GetLhsMeta() == other.GetLhsMeta() &&
         GetRhsMeta() == other.GetRhsMeta() &&
         GetOutputMeta() == other.GetOutputMeta() &&
         dilations_ == other.dilations_ && group_ == other.group_ &&
         kernel_shape_ == other.kernel_shape_ && pads_ == other.pads_ &&
         strides_ == other.strides_ && bias_ == other.bias_;
}

} // namespace kernel
} // namespace fluidml
