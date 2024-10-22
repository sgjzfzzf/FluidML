#ifndef CPU_TRANSFORMERS_KERNEL_GENERATOR_NEG_H_
#define CPU_TRANSFORMERS_KERNEL_GENERATOR_NEG_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/neg.h"

namespace cpu_transformers {
namespace kernel {

class NegKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~NegKernelGenerator() = default;
  virtual std::shared_ptr<NegKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<NegKernelGenerator> Make(Meta &&input_meta,
                                                  Meta &&output_meta);

protected:
  NegKernelGenerator() = default;
  NegKernelGenerator(const NegKernelGenerator &) = delete;
  NegKernelGenerator(NegKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif