#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_SOFTMAX_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_SOFTMAX_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/softmax.h"

namespace cpu_transformers {
namespace kernel {

class SoftmaxKernelGenerator : public SingleInputWithBufferKernelGenerator {
public:
  virtual ~SoftmaxKernelGenerator() = default;
  virtual std::shared_ptr<SoftmaxKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<SoftmaxKernelGenerator> Make(int64_t axis);

protected:
  SoftmaxKernelGenerator() = default;
  SoftmaxKernelGenerator(const SoftmaxKernelGenerator &) = delete;
  SoftmaxKernelGenerator(SoftmaxKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
