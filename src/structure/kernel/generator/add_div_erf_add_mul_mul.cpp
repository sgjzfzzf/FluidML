#include "structure/kernel/generator/add_div_erf_add_mul_mul.h"
#include "structure/tensor/meta.h"

namespace cpu_transformers {
namespace kernel {
class AddDivErfAddMulMulKernelGeneratorImpl
    : public AddDivErfAddMulMulKernelGenerator {
public:
  AddDivErfAddMulMulKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                                        Tensor &&add0_weight, Type div_type,
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
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Tensor add0_weight_;
  const Type div_type_;
  const float64_t div_weight_;
  const Type add1_type_;
  const float64_t add1_weight_;
  const Type mul1_type_;
  const float64_t mul1_weight_;
};

std::unique_ptr<AddDivErfAddMulMulKernelGenerator>
AddDivErfAddMulMulKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                                        Tensor &&add0_weight, Type div_type,
                                        float64_t div_weight, Type add1_type,
                                        float64_t add1_weight, Type mul1_type,
                                        float64_t mul1_weight) {
  return std::make_unique<AddDivErfAddMulMulKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(add0_weight),
      div_type, div_weight, add1_type, add1_weight, mul1_type, mul1_weight);
}

AddDivErfAddMulMulKernelGeneratorImpl::AddDivErfAddMulMulKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, Tensor &&add0_weight, Type div_type,
    float64_t div_weight, Type add1_type, float64_t add1_weight, Type mul1_type,
    float64_t mul1_weight)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      add0_weight_(std::move(add0_weight)), div_type_(div_type),
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

const Meta &AddDivErfAddMulMulKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &AddDivErfAddMulMulKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string AddDivErfAddMulMulKernelGeneratorImpl::GetKernelName() const {
  return AddDivErfAddMulMulKernel::kKernelName;
}

} // namespace kernel
} // namespace cpu_transformers