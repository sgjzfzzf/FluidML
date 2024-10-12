#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_MATMUL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_MATMUL_H_

#include "structure/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

class MatMulKernel : public DoubleInputsWithoutBufferKernel {
public:
  MatMulKernel() = default;
  MatMulKernel(const MatMulKernel &other) = delete;
  MatMulKernel(MatMulKernel &&other) = default;
  virtual ~MatMulKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "MatMulKernel";
};

} // namespace kernel
} // namespace cpu_transformers

#endif
