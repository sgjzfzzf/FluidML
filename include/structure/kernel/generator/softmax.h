#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_SOFTMAX_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_SOFTMAX_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/softmax.h"
#include "structure/tensor/meta.h"

namespace fluidml {
namespace kernel {

class SoftmaxKernelGenerator : public SingleInputWithBufferKernelGenerator {
public:
  virtual ~SoftmaxKernelGenerator() = default;
  virtual std::shared_ptr<SoftmaxKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<SoftmaxKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, int64_t axis);

protected:
  SoftmaxKernelGenerator() = default;
  SoftmaxKernelGenerator(const SoftmaxKernelGenerator &) = delete;
  SoftmaxKernelGenerator(SoftmaxKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
