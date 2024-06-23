#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GATHER_ADD_ADD_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GATHER_ADD_ADD_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace cpu_transformers {
namespace kernel {

class GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel : public Kernel {
public:
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel() = default;
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel(
      const GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel
          &gather_kernel) = delete;
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel(
      GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel &&gather_kernel) =
      default;
  ~GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel() = default;
  void Run(mlir::OpBuilder &builder, const Tensor &data,
           const Tensor &add0_weight, const Tensor &add1_weight,
           mlir::Value &input, mlir::Value &output);
};

} // namespace kernel
} // namespace cpu_transformers

#endif
