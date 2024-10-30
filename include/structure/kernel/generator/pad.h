#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_PAD_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_PAD_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/pad.h"

namespace cpu_transformers {
namespace kernel {

class PadKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~PadKernelGenerator() = default;
  virtual std::shared_ptr<PadKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<PadKernelGenerator>
  Make(Meta &&lhs_meta, Meta &&output_meta,
       std::vector<std::tuple<int64_t, int64_t>> &&pads);

protected:
  PadKernelGenerator() = default;
  PadKernelGenerator(const PadKernelGenerator &) = delete;
  PadKernelGenerator(PadKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
