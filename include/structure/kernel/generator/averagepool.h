#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_AVERAGEPOOL_H
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_AVERAGEPOOL_H

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/averagepool.h"

namespace fluidml {
namespace kernel {

class AveragePoolWithoutPaddingKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~AveragePoolWithoutPaddingKernelGenerator() = default;
  virtual std::shared_ptr<AveragePoolWithoutPaddingKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<AveragePoolWithoutPaddingKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&dilations,
       std::vector<int64_t> &&kernel_shape, std::vector<int64_t> &&strides);

protected:
  AveragePoolWithoutPaddingKernelGenerator() = default;
  AveragePoolWithoutPaddingKernelGenerator(
      const AveragePoolWithoutPaddingKernelGenerator &) = delete;
  AveragePoolWithoutPaddingKernelGenerator(
      AveragePoolWithoutPaddingKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
