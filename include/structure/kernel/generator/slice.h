#ifndef CPU_TRANSFORMERS_KERNEL_GENERATOR_SLICE_H_
#define CPU_TRANSFORMERS_KERNEL_GENERATOR_SLICE_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/slice.h"

namespace cpu_transformers {
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
  SliceKernelGenerator(const SliceKernelGenerator &generator) = delete;
  SliceKernelGenerator(SliceKernelGenerator &&generator) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
