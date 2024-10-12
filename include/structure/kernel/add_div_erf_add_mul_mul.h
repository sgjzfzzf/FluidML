#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_ADD_DIV_ERF_ADD_MUL_MUL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_ADD_DIV_ERF_ADD_MUL_MUL_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace cpu_transformers {
namespace kernel {

class AddDivErfAddMulMulKernel : public SingleInputWithoutBufferKernel {
public:
  AddDivErfAddMulMulKernel(Tensor &&add0_weight, Type div_type,
                           float64_t div_weight, Type add1_type,
                           float64_t add1_weight, Type mul1_type,
                           float64_t mul1_weight);
  AddDivErfAddMulMulKernel(
      const AddDivErfAddMulMulKernel &add_div_erf_add_mul_mul_kernel) = delete;
  AddDivErfAddMulMulKernel(
      AddDivErfAddMulMulKernel &&add_div_erf_add_mul_mul_kernel) = default;
  virtual ~AddDivErfAddMulMulKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "AddDivErfAddMulMulKernel";
  const Tensor add0_weight_;
  const Type div_type_;
  const float64_t div_weight_;
  const Type add1_type_;
  const float64_t add1_weight_;
  const Type mul1_type_;
  const float64_t mul1_weight_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
