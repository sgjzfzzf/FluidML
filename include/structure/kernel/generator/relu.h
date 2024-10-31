#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_RELU_H
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_RELU_H

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/relu.h"

namespace cpu_transformers {
namespace kernel {

class ReluKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~ReluKernelGenerator() = default;
  virtual std::shared_ptr<ReluKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<ReluKernelGenerator> Make(Meta &&input_meta,
                                                   Meta &&output_meta);

protected:
  ReluKernelGenerator() = default;
  ReluKernelGenerator(const ReluKernelGenerator &) = delete;
  ReluKernelGenerator(ReluKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
