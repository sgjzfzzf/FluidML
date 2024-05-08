#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_SUB_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_SUB_H_

#include "structure/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {
class SubConstantScalarLhsKernel : public Kernel {
public:
  SubConstantScalarLhsKernel(Type type, float64_t value);
  SubConstantScalarLhsKernel(const SubConstantScalarLhsKernel &sub_kernel) =
      delete;
  SubConstantScalarLhsKernel(SubConstantScalarLhsKernel &&sub_kernel) = default;
  ~SubConstantScalarLhsKernel() = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &input, mlir::Value &output);

protected:
  Type type_;
  float64_t value_;
};
} // namespace kernel
} // namespace cpu_transformers

#endif
