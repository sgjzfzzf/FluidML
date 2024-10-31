#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_ERF_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_ERF_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/erf.h"
#include "structure/tensor/meta.h"

namespace fluidml {
namespace kernel {

class ErfKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~ErfKernelGenerator() = default;
  virtual std::shared_ptr<ErfKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<ErfKernelGenerator> Make(Meta &&input_meta,
                                                  Meta &&output_meta);

protected:
  ErfKernelGenerator() = default;
  ErfKernelGenerator(const ErfKernelGenerator &) = delete;
  ErfKernelGenerator(ErfKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
