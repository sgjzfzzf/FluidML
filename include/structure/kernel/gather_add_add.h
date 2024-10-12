#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GATHER_ADD_ADD_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GATHER_ADD_ADD_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace cpu_transformers {
namespace kernel {

class GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel
    : public SingleInputWithoutBufferKernel {
public:
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel(Tensor &&data,
                                                         Tensor &&add0_weight,
                                                         Tensor &&add1_weight);
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel(
      const GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel
          &gather_kernel) = delete;
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel(
      GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel &&gather_kernel) =
      default;
  virtual ~GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] =
      "GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel";
  Tensor data_;
  Tensor add0_weight_;
  Tensor add1_weight_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
