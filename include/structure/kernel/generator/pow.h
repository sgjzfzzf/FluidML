#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_POW_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_POW_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/pow.h"

namespace cpu_transformers {
namespace kernel {

class PowKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~PowKernelGenerator() = default;
  virtual std::shared_ptr<PowKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<PowKernelGenerator> Make(Type type, float64_t exp);

protected:
  PowKernelGenerator() = default;
  PowKernelGenerator(const PowKernelGenerator &generator) = delete;
  PowKernelGenerator(PowKernelGenerator &&generator) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
