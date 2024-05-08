#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GEMM_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GEMM_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace cpu_transformers {
namespace kernel {
class GemmConstantWeightsBiasKernel : public Kernel {
public:
  GemmConstantWeightsBiasKernel(float64_t alpha, float64_t beta, bool transA,
                                bool transB);
  GemmConstantWeightsBiasKernel(const GemmConstantWeightsBiasKernel &other) =
      delete;
  GemmConstantWeightsBiasKernel(GemmConstantWeightsBiasKernel &&other) =
      default;
  void Run(mlir::OpBuilder &builder, mlir::Value input, const Tensor &weights,
           const Tensor &bias, mlir::Value output);

private:
  const float64_t alpha_;
  const float64_t beta_;
  const bool transA_;
  const bool transB_;
};
} // namespace kernel
} // namespace cpu_transformers

#endif
