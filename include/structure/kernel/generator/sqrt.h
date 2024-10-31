#ifndef FLUIDML_KERNEL_GENERATOR_SQRT_H_
#define FLUIDML_KERNEL_GENERATOR_SQRT_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/sqrt.h"

namespace fluidml {
namespace kernel {

class SqrtKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~SqrtKernelGenerator() = default;
  virtual std::shared_ptr<SqrtKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<SqrtKernelGenerator> Make(Meta &&input_meta,
                                                   Meta &&output_meta);

protected:
  SqrtKernelGenerator() = default;
  SqrtKernelGenerator(const SqrtKernelGenerator &) = delete;
  SqrtKernelGenerator(SqrtKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
