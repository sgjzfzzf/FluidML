#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_DROPOUT_H
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_DROPOUT_H

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/dropout.h"
#include "utils/float.h"

namespace cpu_transformers {
namespace kernel {

class DropoutKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~DropoutKernelGenerator() = default;
  virtual std::shared_ptr<DropoutKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<DropoutKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, float64_t ratio);

protected:
  DropoutKernelGenerator() = default;
  DropoutKernelGenerator(const DropoutKernelGenerator &) = delete;
  DropoutKernelGenerator(DropoutKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
