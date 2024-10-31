#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_CUM_SUM_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_CUM_SUM_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/cum_sum.h"

namespace fluidml {
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
} // namespace fluidml

#endif
