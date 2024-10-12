#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_SUB_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_SUB_H_

#include "structure/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class SubConstantScalarLhsKernel : public SingleInputWithoutBufferKernel {
public:
  SubConstantScalarLhsKernel(Type type, float64_t value);
  SubConstantScalarLhsKernel(const SubConstantScalarLhsKernel &sub_kernel) =
      delete;
  SubConstantScalarLhsKernel(SubConstantScalarLhsKernel &&sub_kernel) = default;
  virtual ~SubConstantScalarLhsKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

protected:
  static constexpr char kKernelName[] = "SubConstantScalarLhsKernel";
  Type type_;
  float64_t value_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
