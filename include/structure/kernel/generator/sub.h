#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_SUB_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_SUB_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/sub.h"
#include "structure/tensor/meta.h"

namespace cpu_transformers {
namespace kernel {

class SubConstantLhsKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~SubConstantLhsKernelGenerator() = default;
  virtual std::shared_ptr<SubConstantLhsKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<SubConstantLhsKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, Type type, float64_t value);

protected:
  SubConstantLhsKernelGenerator() = default;
  SubConstantLhsKernelGenerator(const SubConstantLhsKernelGenerator &) = delete;
  SubConstantLhsKernelGenerator(SubConstantLhsKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
