#ifndef FLUIDML_KERNEL_GENERATOR_REDUCE_MEAN_H_
#define FLUIDML_KERNEL_GENERATOR_REDUCE_MEAN_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/reduce_mean.h"

namespace fluidml {
namespace kernel {

class ReduceMeanKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~ReduceMeanKernelGenerator() = default;
  virtual std::shared_ptr<ReduceMeanKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<ReduceMeanKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, llvm::SmallVector<int64_t> &&axes,
       bool keep_dims);

protected:
  ReduceMeanKernelGenerator() = default;
  ReduceMeanKernelGenerator(const ReduceMeanKernelGenerator &) = delete;
  ReduceMeanKernelGenerator(ReduceMeanKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
