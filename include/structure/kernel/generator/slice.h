#ifndef FLUIDML_KERNEL_GENERATOR_SLICE_H_
#define FLUIDML_KERNEL_GENERATOR_SLICE_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/slice.h"

namespace fluidml {
namespace kernel {

class SliceKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~SliceKernelGenerator() = default;
  virtual std::shared_ptr<SliceKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<SliceKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta,
       llvm::SmallVector<llvm::SmallVector<int64_t, 4>> &&informations);

protected:
  SliceKernelGenerator() = default;
  SliceKernelGenerator(const SliceKernelGenerator &) = delete;
  SliceKernelGenerator(SliceKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
