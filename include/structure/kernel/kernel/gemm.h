#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GEMM_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GEMM_H_

#include "structure/kernel/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace cpu_transformers {
namespace kernel {

class GemmConstantWeightsBiasKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "GemmConstantWeightsBiasKernel";
  GemmConstantWeightsBiasKernel(float64_t alpha, float64_t beta, bool transA,
                                bool transB, Tensor &&weights, Tensor &&bias);
  GemmConstantWeightsBiasKernel(const GemmConstantWeightsBiasKernel &other) =
      delete;
  GemmConstantWeightsBiasKernel(GemmConstantWeightsBiasKernel &&other) =
      default;
  virtual ~GemmConstantWeightsBiasKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const float64_t alpha_;
  const float64_t beta_;
  const bool transA_;
  const bool transB_;
  const Tensor weights_;
  const Tensor bias_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
