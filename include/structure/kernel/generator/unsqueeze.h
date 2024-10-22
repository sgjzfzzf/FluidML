#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_UNSQUEEZE_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_UNSQUEEZE_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/unsqueeze.h"
#include "structure/tensor/meta.h"

namespace cpu_transformers {
namespace kernel {

class UnSqueezeKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~UnSqueezeKernelGenerator() = default;
  virtual std::shared_ptr<UnSqueezeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<UnSqueezeKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&axes);

protected:
  UnSqueezeKernelGenerator() = default;
  UnSqueezeKernelGenerator(const UnSqueezeKernelGenerator &) = delete;
  UnSqueezeKernelGenerator(UnSqueezeKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
