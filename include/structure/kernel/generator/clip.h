#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_CLIP_H
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_CLIP_H

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/clip.h"

namespace fluidml {
namespace kernel {

class ClipKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~ClipKernelGenerator() = default;
  virtual std::shared_ptr<ClipKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<ClipKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, float32_t min, float32_t max);

protected:
  ClipKernelGenerator() = default;
  ClipKernelGenerator(const ClipKernelGenerator &) = delete;
  ClipKernelGenerator(ClipKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
