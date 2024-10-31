#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_GATHER_ADD_ADD_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_GATHER_ADD_ADD_H_

#include "structure/kernel/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace fluidml {
namespace kernel {

class GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel
    : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] =
      "GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel";
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel(Tensor &&data,
                                                         Tensor &&add0_weight,
                                                         Tensor &&add1_weight);
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel(
      const GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel &) = delete;
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel(
      GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel &&) = default;
  virtual ~GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const Tensor data_;
  const Tensor add0_weight_;
  const Tensor add1_weight_;
};

} // namespace kernel
} // namespace fluidml

#endif
