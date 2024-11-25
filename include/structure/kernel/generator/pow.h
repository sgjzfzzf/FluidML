#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_POW_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_POW_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/pow.h"
#include "structure/tensor/meta.h"

namespace fluidml {
namespace kernel {

class PowKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~PowKernelGenerator() = default;
  virtual std::shared_ptr<PowKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<PowKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, Type type, float64_t exp);

protected:
  PowKernelGenerator() = default;
  PowKernelGenerator(const PowKernelGenerator &) = delete;
  PowKernelGenerator(PowKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
