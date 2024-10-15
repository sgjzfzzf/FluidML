#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_GEMM_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_GEMM_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/gemm.h"

namespace cpu_transformers {
namespace kernel {

class GemmConstantWeightsBiasKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~GemmConstantWeightsBiasKernelGenerator() = default;
  virtual std::shared_ptr<GemmConstantWeightsBiasKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<GemmConstantWeightsBiasKernelGenerator>
  Make(float64_t alpha, float64_t beta, bool transA, bool transB,
       Tensor &&weights, Tensor &&bias);

protected:
  GemmConstantWeightsBiasKernelGenerator() = default;
  GemmConstantWeightsBiasKernelGenerator(
      const GemmConstantWeightsBiasKernelGenerator &generator) = delete;
  GemmConstantWeightsBiasKernelGenerator(
      GemmConstantWeightsBiasKernelGenerator &&generator) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
