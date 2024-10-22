#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_MATMUL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_MATMUL_H_

#include "structure/kernel/kernel/kernel.h"
#include "structure/kernel/kernel/utils.h"

namespace cpu_transformers {
namespace kernel {

class MatMulKernel : public DoubleInputsWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "MatMulKernel";
  MatMulKernel();
  MatMulKernel(llvm::SmallVector<Axis, 3> &&axes);
  MatMulKernel(const MatMulKernel &) = delete;
  MatMulKernel(MatMulKernel &&) = default;
  virtual ~MatMulKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;

private:
  llvm::SmallVector<Axis> axes_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
