#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_SQUEEZE_H
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_SQUEEZE_H

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/squeeze.h"

namespace fluidml {
namespace kernel {

class SqueezeKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~SqueezeKernelGenerator() = default;
  virtual std::shared_ptr<SqueezeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<SqueezeKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&axes);

protected:
  SqueezeKernelGenerator() = default;
  SqueezeKernelGenerator(const SqueezeKernelGenerator &) = delete;
  SqueezeKernelGenerator(SqueezeKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
