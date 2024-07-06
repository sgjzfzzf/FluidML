#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_MUL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_MUL_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"
#include "utils/float.h"

namespace cpu_transformers {
namespace kernel {
class MulKernelInterface : virtual public Kernel {
public:
  MulKernelInterface() = default;
  MulKernelInterface(const MulKernelInterface &other) = delete;
  MulKernelInterface(MulKernelInterface &&other) = default;
};

class MulConstantScalarKernel : public SingleInputWithoutBufferKernel,
                                public MulKernelInterface {
public:
  MulConstantScalarKernel(Type type, float64_t constant);
  MulConstantScalarKernel(const MulConstantScalarKernel &other) = delete;
  MulConstantScalarKernel(MulConstantScalarKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs,
           mlir::Value &output) const;

private:
  const Type type_;
  const float64_t constant_;
};

class MulConstantTensorKernel : public SingleInputWithoutBufferKernel,
                                public MulKernelInterface {
public:
  MulConstantTensorKernel(const Tensor &constant);
  MulConstantTensorKernel(const MulConstantTensorKernel &other) = delete;
  MulConstantTensorKernel(MulConstantTensorKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs,
           mlir::Value &output) const;

private:
  const Tensor constant_;
};

class MulCommonKernel : public DoubleInputsWithoutBufferKernel,
                        public MulKernelInterface {
public:
  MulCommonKernel() = default;
  MulCommonKernel(const MulCommonKernel &other) = delete;
  MulCommonKernel(MulCommonKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
