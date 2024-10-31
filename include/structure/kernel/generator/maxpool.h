#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_MAXPOOL_H
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_MAXPOOL_H

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/maxpool.h"
#include <cstdint>

namespace cpu_transformers {
namespace kernel {

class MaxPoolWithoutPaddingKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~MaxPoolWithoutPaddingKernelGenerator() = default;
  virtual std::shared_ptr<MaxPoolWithoutPaddingKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<MaxPoolWithoutPaddingKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> kernel_shape,
       std::vector<int64_t> strides);

protected:
  MaxPoolWithoutPaddingKernelGenerator() = default;
  MaxPoolWithoutPaddingKernelGenerator(
      const MaxPoolWithoutPaddingKernelGenerator &) = delete;
  MaxPoolWithoutPaddingKernelGenerator(
      MaxPoolWithoutPaddingKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
