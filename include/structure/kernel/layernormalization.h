#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_LAYERNORMALIZATION_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_LAYERNORMALIZATION_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"
#include "utils/float.h"

namespace cpu_transformers {
namespace kernel {
class LayerNormalizationConstantScaleBiasKernel : public Kernel {
public:
  LayerNormalizationConstantScaleBiasKernel(int64_t axis, float64_t epsilon);
  LayerNormalizationConstantScaleBiasKernel(
      const LayerNormalizationConstantScaleBiasKernel &other) = delete;
  LayerNormalizationConstantScaleBiasKernel(
      LayerNormalizationConstantScaleBiasKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &input, const Tensor &scale,
           const Tensor &bias, mlir::Value &output, mlir::Value &buffer);

private:
  int64_t axis_;
  float64_t epsilon_;
};
} // namespace kernel
} // namespace cpu_transformers

#endif
