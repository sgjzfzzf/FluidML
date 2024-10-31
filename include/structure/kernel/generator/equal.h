#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_EQUAL_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_EQUAL_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/equal.h"

namespace fluidml {
namespace kernel {

class EqualKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~EqualKernelGenerator() = default;
  virtual std::shared_ptr<EqualKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<EqualKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, Type type, float64_t value);

protected:
  EqualKernelGenerator() = default;
  EqualKernelGenerator(const EqualKernelGenerator &) = delete;
  EqualKernelGenerator(EqualKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
