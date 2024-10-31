#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_CAST_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_CAST_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/cast.h"

namespace fluidml {
namespace kernel {

class CastKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~CastKernelGenerator() = default;
  virtual std::shared_ptr<CastKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<CastKernelGenerator> Make(Meta &&input_meta,
                                                   Meta &&output_meta);

protected:
  CastKernelGenerator() = default;
  CastKernelGenerator(const CastKernelGenerator &) = delete;
  CastKernelGenerator(CastKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
