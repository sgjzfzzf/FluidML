#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_DIV_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_DIV_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/div.h"

namespace cpu_transformers {
namespace kernel {

class DivConstantRhsKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~DivConstantRhsKernelGenerator() = default;
  virtual std::shared_ptr<DivConstantRhsKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<SingleInputWithoutBufferKernelGenerator>
  Make(Type type, float64_t constant);

protected:
  DivConstantRhsKernelGenerator() = default;
  DivConstantRhsKernelGenerator(
      const DivConstantRhsKernelGenerator &generator) = delete;
  DivConstantRhsKernelGenerator(DivConstantRhsKernelGenerator &&generator) =
      default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif