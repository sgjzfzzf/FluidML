#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_TANH_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_TANH_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/tanh.h"

namespace cpu_transformers {
namespace kernel {

class TanhKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~TanhKernelGenerator() = default;
  virtual std::shared_ptr<TanhKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<TanhKernelGenerator> Make();

protected:
  TanhKernelGenerator() = default;
  TanhKernelGenerator(const TanhKernelGenerator &) = delete;
  TanhKernelGenerator(TanhKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
