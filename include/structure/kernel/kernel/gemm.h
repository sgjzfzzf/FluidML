#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_GEMM_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_GEMM_H_

#include "structure/kernel/kernel/kernel.h"
#include "structure/kernel/kernel/utils.h"
#include "structure/tensor/tensor.h"

namespace cpu_transformers {
namespace kernel {

class GemmConstantBiasKernel : public DoubleInputsWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "GemmConstantWeightsBiasKernel";
  GemmConstantBiasKernel(float64_t alpha, float64_t beta, bool transA,
                         bool transB, Tensor &&bias);
  GemmConstantBiasKernel(float64_t alpha, float64_t beta, bool transA,
                         bool transB, Tensor &&bias,
                         llvm::SmallVector<Axis, 3> &&axes);
  GemmConstantBiasKernel(const GemmConstantBiasKernel &) = delete;
  GemmConstantBiasKernel(GemmConstantBiasKernel &&) = default;
  virtual ~GemmConstantBiasKernel() = default;
  std::string GetKernelName() const override;
  llvm::ArrayRef<Axis> GetAxes();
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;

private:
  const float64_t alpha_;
  const float64_t beta_;
  const bool transA_;
  const bool transB_;
  const Tensor bias_;
  llvm::SmallVector<Axis, 3> axes_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
