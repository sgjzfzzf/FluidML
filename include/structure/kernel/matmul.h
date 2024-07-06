#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_MATMUL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_MATMUL_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace cpu_transformers {
namespace kernel {

class MatMulKernel : virtual public Kernel {
public:
  MatMulKernel() = default;
  MatMulKernel(const MatMulKernel &other) = delete;
  MatMulKernel(MatMulKernel &&other) = default;

protected:
  void run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const;
};

class MatMulConstantLhsKernel : public SingleInputWithoutBufferKernel,
                                public MatMulKernel {
public:
  MatMulConstantLhsKernel(Tensor &&weight);
  MatMulConstantLhsKernel(const MatMulConstantLhsKernel &other) = delete;
  MatMulConstantLhsKernel(MatMulConstantLhsKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  Tensor weight_;
};

class MatMulConstantRhsKernel : public SingleInputWithoutBufferKernel,
                                public MatMulKernel {
public:
  MatMulConstantRhsKernel(Tensor &&weight);
  MatMulConstantRhsKernel(const MatMulConstantRhsKernel &other) = delete;
  MatMulConstantRhsKernel(MatMulConstantRhsKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  Tensor weight_;
};

class MatMulCommonKernel : public DoubleInputsWithoutBufferKernel,
                           public MatMulKernel {
public:
  MatMulCommonKernel() = default;
  MatMulCommonKernel(const MatMulCommonKernel &other) = delete;
  MatMulCommonKernel(MatMulCommonKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
