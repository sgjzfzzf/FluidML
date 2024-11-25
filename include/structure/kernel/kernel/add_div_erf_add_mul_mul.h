#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_KERNEL_ADD_DIV_ERF_ADD_MUL_MUL_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_KERNEL_ADD_DIV_ERF_ADD_MUL_MUL_H_

#include "structure/kernel/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace fluidml {
namespace kernel {

class AddDivErfAddMulMulKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "AddDivErfAddMulMulKernel";
  AddDivErfAddMulMulKernel(Tensor &&add0_weight, Type div_type,
                           float64_t div_weight, Type add1_type,
                           float64_t add1_weight, Type mul1_type,
                           float64_t mul1_weight);
  AddDivErfAddMulMulKernel(const AddDivErfAddMulMulKernel &) = delete;
  AddDivErfAddMulMulKernel(AddDivErfAddMulMulKernel &&) = default;
  virtual ~AddDivErfAddMulMulKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const Tensor add0_weight_;
  const Type div_type_;
  const float64_t div_weight_;
  const Type add1_type_;
  const float64_t add1_weight_;
  const Type mul1_type_;
  const float64_t mul1_weight_;
};

} // namespace kernel
} // namespace fluidml

#endif
