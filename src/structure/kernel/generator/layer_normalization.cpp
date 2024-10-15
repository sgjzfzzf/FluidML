#include "structure/kernel/generator/layer_normalization.h"

namespace cpu_transformers {
namespace kernel {

class LayerNormalizationConstantScaleBiasKernelGeneratorImpl
    : public LayerNormalizationConstantScaleBiasKernelGenerator {
public:
  LayerNormalizationConstantScaleBiasKernelGeneratorImpl(int64_t axis,
                                                         float64_t epsilon,
                                                         Tensor &&scale,
                                                         Tensor &&bias);
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

private:
  const int64_t axis_;
  const float64_t epsilon_;
  const Tensor scale_;
  const Tensor bias_;
};

std::unique_ptr<LayerNormalizationConstantScaleBiasKernelGenerator>
LayerNormalizationConstantScaleBiasKernelGenerator::Make(int64_t axis,
                                                         float64_t epsilon,
                                                         Tensor &&scale,
                                                         Tensor &&bias) {
  return std::make_unique<
      LayerNormalizationConstantScaleBiasKernelGeneratorImpl>(
      axis, epsilon, std::move(scale), std::move(bias));
}

LayerNormalizationConstantScaleBiasKernelGeneratorImpl::
    LayerNormalizationConstantScaleBiasKernelGeneratorImpl(int64_t axis,
                                                           float64_t epsilon,
                                                           Tensor &&scale,
                                                           Tensor &&bias)
    : axis_(axis), epsilon_(epsilon), scale_(std::move(scale)),
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

} // namespace kernel
} // namespace cpu_transformers
