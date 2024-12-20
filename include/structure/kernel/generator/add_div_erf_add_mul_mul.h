#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_ADD_DIV_ERF_ADD_MUL_MUL_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_ADD_DIV_ERF_ADD_MUL_MUL_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/add_div_erf_add_mul_mul.h"

namespace fluidml {
namespace kernel {

class AddDivErfAddMulMulKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~AddDivErfAddMulMulKernelGenerator() = default;
  virtual std::shared_ptr<AddDivErfAddMulMulKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<AddDivErfAddMulMulKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, Tensor &&add0_weight,
       Type div_type, float64_t div_weight, Type add1_type,
       float64_t add1_weight, Type mul1_type, float64_t mul1_weight);

protected:
  AddDivErfAddMulMulKernelGenerator() = default;
  AddDivErfAddMulMulKernelGenerator(const AddDivErfAddMulMulKernelGenerator &) =
      delete;
  AddDivErfAddMulMulKernelGenerator(AddDivErfAddMulMulKernelGenerator &&) =
      default;
};

} // namespace kernel
} // namespace fluidml

#endif