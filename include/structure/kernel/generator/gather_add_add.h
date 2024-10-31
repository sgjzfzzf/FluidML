#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_GATHER_ADD_ADD_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_GATHER_ADD_ADD_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/gather_add_add.h"

namespace fluidml {
namespace kernel {

class GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator() =
      default;
  virtual std::shared_ptr<
      GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<
      GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, Tensor &&data,
       Tensor &&add0_weight, Tensor &&add1_weight);

protected:
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator() = default;
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator(
      const GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator &) =
      delete;
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator(
      GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator &&) =
      default;
};

} // namespace kernel
} // namespace fluidml

#endif
