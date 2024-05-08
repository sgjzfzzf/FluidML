#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_MATMUL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_MATMUL_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace cpu_transformers {
namespace kernel {

class MatMulKernel : public Kernel {
public:
  MatMulKernel() = default;
  MatMulKernel(const MatMulKernel &other) = delete;
  MatMulKernel(MatMulKernel &&other) = default;

protected:
  void run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output);
};

class MatMulConstantLhsKernel : public MatMulKernel {
public:
  MatMulConstantLhsKernel() = default;
  MatMulConstantLhsKernel(const MatMulConstantLhsKernel &other) = delete;
  MatMulConstantLhsKernel(MatMulConstantLhsKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, const Tensor &lhs, mlir::Value &rhs,
           mlir::Value &output);
};

class MatMulConstantRhsKernel : public MatMulKernel {
public:
  MatMulConstantRhsKernel() = default;
  MatMulConstantRhsKernel(const MatMulConstantRhsKernel &other) = delete;
  MatMulConstantRhsKernel(MatMulConstantRhsKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, const Tensor &rhs,
           mlir::Value &output);
};

class MatMulCommonKernel : public MatMulKernel {
public:
  MatMulCommonKernel() = default;
  MatMulCommonKernel(const MatMulCommonKernel &other) = delete;
  MatMulCommonKernel(MatMulCommonKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output);
};

} // namespace kernel
} // namespace cpu_transformers

#endif
