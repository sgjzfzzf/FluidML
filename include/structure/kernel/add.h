#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_ADD_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_ADD_H_

#include "mlir/IR/Builders.h"
#include "structure/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class AddConstantScalarKernel : public SingleInputWithoutBufferKernel {
public:
  AddConstantScalarKernel(Type type, float64_t constant);
  AddConstantScalarKernel(const AddConstantScalarKernel &add_kernel) = delete;
  AddConstantScalarKernel(AddConstantScalarKernel &&add_kernel) = default;
  ~AddConstantScalarKernel() = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  Type type_;
  float64_t constant_;
};

class AddCommonKernel : public DoubleInputsWithoutBufferKernel {
public:
  AddCommonKernel() = default;
  AddCommonKernel(const AddCommonKernel &add_kernel) = delete;
  AddCommonKernel(AddCommonKernel &&add_kernel) = default;
  ~AddCommonKernel() = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
