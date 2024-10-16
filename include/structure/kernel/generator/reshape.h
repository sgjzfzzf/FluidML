#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_RESHAPE_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_RESHAPE_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/reshape.h"
#include "structure/tensor/meta.h"

namespace cpu_transformers {
namespace kernel {

class ReshapeKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~ReshapeKernelGenerator() = default;
  virtual std::shared_ptr<ReshapeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<ReshapeKernelGenerator> Make(Meta &&input_meta,
                                                      Meta &&output_meta);

protected:
  ReshapeKernelGenerator() = default;
  ReshapeKernelGenerator(const ReshapeKernelGenerator &) = delete;
  ReshapeKernelGenerator(ReshapeKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
