#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_DIV_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_DIV_H_

#include "structure/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class DivConstantRhsKernel : public SingleInputWithoutBufferKernel {
public:
  DivConstantRhsKernel(Type type, float64_t constant);
  DivConstantRhsKernel(const DivConstantRhsKernel &div_kernel) = delete;
  DivConstantRhsKernel(DivConstantRhsKernel &&div_kernel) = default;
  virtual ~DivConstantRhsKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "DivConstantRhsKernel";
  Type type_;
  float64_t constant_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif