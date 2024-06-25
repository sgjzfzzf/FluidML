#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_UNSQUEEZE_SUB_MUL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_UNSQUEEZE_SUB_MUL_H_

#include "structure/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class UnsqueezeSubLhsScalarMulRhsScalarKernel : public Kernel {
public:
  UnsqueezeSubLhsScalarMulRhsScalarKernel() = default;
  UnsqueezeSubLhsScalarMulRhsScalarKernel(
      const UnsqueezeSubLhsScalarMulRhsScalarKernel &other) = delete;
  UnsqueezeSubLhsScalarMulRhsScalarKernel(
      UnsqueezeSubLhsScalarMulRhsScalarKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, llvm::ArrayRef<int64_t> unsqueeze_axes,
           const Type &sub_type, float64_t sub_val, const Type &mul_type,
           float64_t mul_val, mlir::Value &input, mlir::Value &output);
};

} // namespace kernel
} // namespace cpu_transformers

#endif
