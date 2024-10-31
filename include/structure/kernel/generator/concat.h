#ifndef FLUIDML_KERNEL_GENERATOR_CONCAT_H_
#define FLUIDML_KERNEL_GENERATOR_CONCAT_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/concat.h"

namespace fluidml {
namespace kernel {

class Concat2KernelGenerator : public DoubleInputsWithoutBufferKernelGenerator {
public:
  virtual ~Concat2KernelGenerator() = default;
  virtual std::shared_ptr<Concat2Kernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<Concat2KernelGenerator>
  Make(Meta &&lhs_meta, Meta &&rhs_meta, Meta &&output_meta, size_t axis);

protected:
  Concat2KernelGenerator() = default;
  Concat2KernelGenerator(const Concat2KernelGenerator &) = delete;
  Concat2KernelGenerator(Concat2KernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
