#include "structure/kernel/generator/layer_normalization.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class LayerNormalizationConstantScaleBiasKernelGeneratorImpl
    : public LayerNormalizationConstantScaleBiasKernelGenerator {
public:
  LayerNormalizationConstantScaleBiasKernelGeneratorImpl(
      Meta &&input_meta, Meta &&output_meta, int64_t axis, float64_t epsilon,
      Tensor &&scale, Tensor &&bias);
  LayerNormalizationConstantScaleBiasKernelGeneratorImpl(
      const LayerNormalizationConstantScaleBiasKernelGeneratorImpl &generator) =
      delete;
  LayerNormalizationConstantScaleBiasKernelGeneratorImpl(
      LayerNormalizationConstantScaleBiasKernelGeneratorImpl &&generator) =
      default;
  virtual ~LayerNormalizationConstantScaleBiasKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithBufferKernel> YieldSingleInputWithBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<LayerNormalizationConstantScaleBiasKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const LayerNormalizationConstantScaleBiasKernelGeneratorImpl
                  &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const int64_t axis_;
  const float64_t epsilon_;
  const Tensor scale_;
  const Tensor bias_;
};

std::unique_ptr<LayerNormalizationConstantScaleBiasKernelGenerator>
LayerNormalizationConstantScaleBiasKernelGenerator::Make(
    Meta &&input_meta, Meta &&output_meta, int64_t axis, float64_t epsilon,
    Tensor &&scale, Tensor &&bias) {
  return std::make_unique<
      LayerNormalizationConstantScaleBiasKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), axis, epsilon,
      std::move(scale), std::move(bias));
}

LayerNormalizationConstantScaleBiasKernelGeneratorImpl::
    LayerNormalizationConstantScaleBiasKernelGeneratorImpl(
        Meta &&input_meta, Meta &&output_meta, int64_t axis, float64_t epsilon,
        Tensor &&scale, Tensor &&bias)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      axis_(axis), epsilon_(epsilon), scale_(std::move(scale)),
      bias_(std::move(bias)) {}

std::shared_ptr<SingleInputWithBufferKernel>
LayerNormalizationConstantScaleBiasKernelGeneratorImpl::
    YieldSingleInputWithBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                     llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<LayerNormalizationConstantScaleBiasKernel>
LayerNormalizationConstantScaleBiasKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  Tensor scale = scale_, bias = bias_;
  return std::make_shared<LayerNormalizationConstantScaleBiasKernel>(
      axis_, epsilon_, std::move(scale), std::move(bias));
}

const Meta &
LayerNormalizationConstantScaleBiasKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &
LayerNormalizationConstantScaleBiasKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string
LayerNormalizationConstantScaleBiasKernelGeneratorImpl::GetKernelName() const {
  return LayerNormalizationConstantScaleBiasKernel::kKernelName;
}

size_t
LayerNormalizationConstantScaleBiasKernelGeneratorImpl::GetHashCode() const {
  std::hash<int64_t> i64_hash;
  std::hash<float64_t> f64_hash;
  size_t hash = input_meta_.GetHashCode();
  hash ^= output_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= i64_hash(axis_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= f64_hash(epsilon_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= scale_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= bias_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool LayerNormalizationConstantScaleBiasKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const LayerNormalizationConstantScaleBiasKernelGeneratorImpl *other_ptr =
          dynamic_cast<
              const LayerNormalizationConstantScaleBiasKernelGeneratorImpl *>(
              &other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool LayerNormalizationConstantScaleBiasKernelGeneratorImpl::Equals(
    const LayerNormalizationConstantScaleBiasKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ &&
         output_meta_ == other.output_meta_ && axis_ == other.axis_ &&
         epsilon_ == other.epsilon_ && scale_ == other.scale_ &&
         bias_ == other.bias_;
}

} // namespace kernel
} // namespace cpu_transformers
