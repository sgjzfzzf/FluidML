#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_CUM_SUM_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_CUM_SUM_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/cum_sum.h"

namespace cpu_transformers {
namespace kernel {

class CumSumKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~CumSumKernelGenerator() = default;
  virtual std::shared_ptr<CumSumKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<CumSumKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, int64_t axis, bool exclusive,
       bool reverse);

protected:
  CumSumKernelGenerator() = default;
  CumSumKernelGenerator(const CumSumKernelGenerator &) = delete;
  CumSumKernelGenerator(CumSumKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
