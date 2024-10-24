#ifndef CPU_TRANSLATION_STRUCTURE_KERNEL_GENERATOR_NOT_H
#define CPU_TRANSLATION_STRUCTURE_KERNEL_GENERATOR_NOT_H

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/not.h"

namespace cpu_transformers {
namespace kernel {

class NotKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~NotKernelGenerator() = default;
  virtual std::shared_ptr<NotKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<NotKernelGenerator> Make(Meta &&input_meta,
                                                  Meta &&output_meta);

protected:
  NotKernelGenerator() = default;
  NotKernelGenerator(const NotKernelGenerator &) = delete;
  NotKernelGenerator(NotKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
