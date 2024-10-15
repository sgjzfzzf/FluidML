#include "structure/kernel/generator/add_div_erf_add_mul_mul.h"

namespace cpu_transformers {
namespace kernel {
class AddDivErfAddMulMulKernelGeneratorImpl
    : public AddDivErfAddMulMulKernelGenerator {
public:
  AddDivErfAddMulMulKernelGeneratorImpl(Tensor &&add0_weight, Type div_type,
                                        float64_t div_weight, Type add1_type,
                                        float64_t add1_weight, Type mul1_type,
                                        float64_t mul1_weight);
  AddDivErfAddMulMulKernelGeneratorImpl(
      const AddDivErfAddMulMulKernelGeneratorImpl &generator) = delete;
  AddDivErfAddMulMulKernelGeneratorImpl(
      AddDivErfAddMulMulKernelGeneratorImpl &&generator) = default;
  virtual ~AddDivErfAddMulMulKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<AddDivErfAddMulMulKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const Tensor add0_weight_;
  const Type div_type_;
  const float64_t div_weight_;
  const Type add1_type_;
  const float64_t add1_weight_;
  const Type mul1_type_;
  const float64_t mul1_weight_;
};

std::unique_ptr<AddDivErfAddMulMulKernelGenerator>
AddDivErfAddMulMulKernelGenerator::Make(Tensor &&add0_weight, Type div_type,
                                        float64_t div_weight, Type add1_type,
                                        float64_t add1_weight, Type mul1_type,
                                        float64_t mul1_weight) {
  return std::make_unique<AddDivErfAddMulMulKernelGeneratorImpl>(
      std::move(add0_weight), div_type, div_weight, add1_type, add1_weight,
      mul1_type, mul1_weight);
}

AddDivErfAddMulMulKernelGeneratorImpl::AddDivErfAddMulMulKernelGeneratorImpl(
    Tensor &&add0_weight, Type div_type, float64_t div_weight, Type add1_type,
    float64_t add1_weight, Type mul1_type, float64_t mul1_weight)
    : add0_weight_(std::move(add0_weight)), div_type_(div_type),
      div_weight_(div_weight), add1_type_(add1_type), add1_weight_(add1_weight),
      mul1_type_(mul1_type), mul1_weight_(mul1_weight) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
AddDivErfAddMulMulKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<AddDivErfAddMulMulKernel>
AddDivErfAddMulMulKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  Tensor add0_weight = add0_weight_;
  return std::make_shared<AddDivErfAddMulMulKernel>(
      std::move(add0_weight), div_type_, div_weight_, add1_type_, add1_weight_,
      mul1_type_, mul1_weight_);
}

} // namespace kernel
} // namespace cpu_transformers