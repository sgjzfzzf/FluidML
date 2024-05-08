#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_MUL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_MUL_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"
#include "utils/float.h"

namespace cpu_transformers {
namespace kernel {
class MulKernel : public Kernel {
public:
  MulKernel() = default;
  MulKernel(const MulKernel &other) = delete;
  MulKernel(MulKernel &&other) = default;
};

class MulConstantScalarKernel : public MulKernel {
public:
  MulConstantScalarKernel(Type type, float64_t constant);
  MulConstantScalarKernel(const MulConstantScalarKernel &other) = delete;
  MulConstantScalarKernel(MulConstantScalarKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &output);

private:
  const Type type_;
  const float64_t constant_;
};

class MulConstantTensorKernel : public MulKernel {
public:
  MulConstantTensorKernel(const Tensor &constant);
  MulConstantTensorKernel(const MulConstantTensorKernel &other) = delete;
  MulConstantTensorKernel(MulConstantTensorKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &output);

private:
  const Tensor constant_;
};

class MulCommonKernel : public MulKernel {
public:
  MulCommonKernel() = default;
  MulCommonKernel(const MulCommonKernel &other) = delete;
  MulCommonKernel(MulCommonKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output);
};

} // namespace kernel
} // namespace cpu_transformers

#endif
