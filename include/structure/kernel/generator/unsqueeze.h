#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_UNSQUEEZE_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_UNSQUEEZE_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/unsqueeze.h"
#include "structure/tensor/meta.h"

namespace fluidml {
namespace kernel {

class UnsqueezeKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~UnsqueezeKernelGenerator() = default;
  virtual std::shared_ptr<UnsqueezeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<UnsqueezeKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&axes);

protected:
  UnsqueezeKernelGenerator() = default;
  UnsqueezeKernelGenerator(const UnsqueezeKernelGenerator &) = delete;
  UnsqueezeKernelGenerator(UnsqueezeKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
