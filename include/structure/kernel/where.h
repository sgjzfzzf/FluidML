#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_WHERE_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_WHERE_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace cpu_transformers {
namespace kernel {

class WhereConstantCondConstantScalarYKernel : public Kernel {
public:
  WhereConstantCondConstantScalarYKernel() = default;
  WhereConstantCondConstantScalarYKernel(
      const WhereConstantCondConstantScalarYKernel &) = delete;
  WhereConstantCondConstantScalarYKernel(
      WhereConstantCondConstantScalarYKernel &&) = default;
  void Run(mlir::OpBuilder &builder, const Tensor &cond, mlir::Value &x,
           Type type, float64_t y, mlir::Value &output);
};

class WhereConstantCondConstantTensorYKernel : public Kernel {
public:
  WhereConstantCondConstantTensorYKernel() = default;
  WhereConstantCondConstantTensorYKernel(
      const WhereConstantCondConstantTensorYKernel &) = delete;
  WhereConstantCondConstantTensorYKernel(
      WhereConstantCondConstantTensorYKernel &&) = default;
  void Run(mlir::OpBuilder &builder, const Tensor &cond, mlir::Value &x,
           const Tensor &y, mlir::Value &output);
};

} // namespace kernel
} // namespace cpu_transformers

#endif
