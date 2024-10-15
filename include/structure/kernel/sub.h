#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_SUB_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_SUB_H_

#include "structure/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class SubConstantLhsKernel : public SingleInputWithoutBufferKernel {
public:
  SubConstantLhsKernel(Type type, float64_t value);
  SubConstantLhsKernel(const SubConstantLhsKernel &sub_kernel) = delete;
  SubConstantLhsKernel(SubConstantLhsKernel &&sub_kernel) = default;
  virtual ~SubConstantLhsKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

protected:
  static constexpr char kKernelName[] = "SubConstantLhsKernel";
  Type type_;
  float64_t value_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
