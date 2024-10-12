#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_LAYER_NORMALIZATION_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_LAYER_NORMALIZATION_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"
#include "utils/float.h"

namespace cpu_transformers {
namespace kernel {
class LayerNormalizationConstantScaleBiasKernel
    : public SingleInputWithBufferKernel {
public:
  LayerNormalizationConstantScaleBiasKernel(int64_t axis, float64_t epsilon,
                                            Tensor &&scale, Tensor &&bias);
  LayerNormalizationConstantScaleBiasKernel(
      const LayerNormalizationConstantScaleBiasKernel &other) = delete;
  LayerNormalizationConstantScaleBiasKernel(
      LayerNormalizationConstantScaleBiasKernel &&other) = default;
  virtual ~LayerNormalizationConstantScaleBiasKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input, mlir::Value &output,
           mlir::Value &buffer) const override;

private:
  static constexpr char kKernelName[] =
      "LayerNormalizationConstantScaleBiasKernel";
  int64_t axis_;
  float64_t epsilon_;
  Tensor scale_;
  Tensor bias_;
};
} // namespace kernel
} // namespace cpu_transformers

#endif
