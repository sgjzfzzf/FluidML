#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_LAYER_NORMALIZATION_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_LAYER_NORMALIZATION_H_

#include "structure/kernel/kernel/kernel.h"
#include "structure/tensor/tensor.h"
#include "utils/float.h"

namespace fluidml {
namespace kernel {

class LayerNormalizationConstantScaleBiasKernel
    : public SingleInputWithBufferKernel {
public:
  static constexpr char kKernelName[] =
      "LayerNormalizationConstantScaleBiasKernel";
  LayerNormalizationConstantScaleBiasKernel(int64_t axis, float64_t epsilon,
                                            Tensor &&scale, Tensor &&bias);
  LayerNormalizationConstantScaleBiasKernel(
      const LayerNormalizationConstantScaleBiasKernel &) = delete;
  LayerNormalizationConstantScaleBiasKernel(
      LayerNormalizationConstantScaleBiasKernel &&) = default;
  virtual ~LayerNormalizationConstantScaleBiasKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input, mlir::Value &output,
           mlir::Value &buffer) const override;

private:
  const int64_t axis_;
  const float64_t epsilon_;
  const Tensor scale_;
  const Tensor bias_;
};

} // namespace kernel
} // namespace fluidml

#endif
