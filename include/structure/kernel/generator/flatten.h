#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_FLATTEN_H
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_FLATTEN_H

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/flatten.h"

namespace fluidml {
namespace kernel {
class FlattenKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~FlattenKernelGenerator() = default;
  virtual std::shared_ptr<FlattenKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<FlattenKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, int64_t axis);

protected:
  FlattenKernelGenerator() = default;
  FlattenKernelGenerator(const FlattenKernelGenerator &) = delete;
  FlattenKernelGenerator(FlattenKernelGenerator &&) = default;
};
} // namespace kernel
} // namespace fluidml

#endif
