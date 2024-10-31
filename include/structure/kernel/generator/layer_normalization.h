#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_LAYER_NORMALIZATION_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_LAYER_NORMALIZATION_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/layer_normalization.h"

namespace fluidml {
namespace kernel {

class LayerNormalizationConstantScaleBiasKernelGenerator
    : public SingleInputWithBufferKernelGenerator {
public:
  virtual ~LayerNormalizationConstantScaleBiasKernelGenerator() = default;
  virtual std::shared_ptr<LayerNormalizationConstantScaleBiasKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<LayerNormalizationConstantScaleBiasKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, int64_t axis, float64_t epsilon,
       Tensor &&scale, Tensor &&bias);

protected:
  LayerNormalizationConstantScaleBiasKernelGenerator() = default;
  LayerNormalizationConstantScaleBiasKernelGenerator(
      const LayerNormalizationConstantScaleBiasKernelGenerator &) = delete;
  LayerNormalizationConstantScaleBiasKernelGenerator(
      LayerNormalizationConstantScaleBiasKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
