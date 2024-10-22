#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_GEMM_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_GEMM_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/gemm.h"
#include "llvm/ADT/ArrayRef.h"

namespace cpu_transformers {
namespace kernel {

class GemmConstantBiasKernelGenerator
    : public DoubleInputsWithoutBufferKernelGenerator {
public:
  virtual ~GemmConstantBiasKernelGenerator() = default;
  virtual std::shared_ptr<GemmConstantBiasKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<GemmConstantBiasKernelGenerator>
  Make(Meta &&lhs_meta, Meta &&rhs_meta, Meta &&output_meta, float64_t alpha,
       float64_t beta, bool transA, bool transB, Tensor &&bias);

protected:
  GemmConstantBiasKernelGenerator() = default;
  GemmConstantBiasKernelGenerator(const GemmConstantBiasKernelGenerator &) =
      delete;
  GemmConstantBiasKernelGenerator(GemmConstantBiasKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
