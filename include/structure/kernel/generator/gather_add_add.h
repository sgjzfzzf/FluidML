#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_GATHER_ADD_ADD_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_GATHER_ADD_ADD_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/gather_add_add.h"

namespace cpu_transformers {
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
  Make(Tensor &&data, Tensor &&add0_weight, Tensor &&add1_weight);

protected:
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator() = default;
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator(
      const GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator
          &generator) = delete;
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator(
      GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator
          &&generator) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
