#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_DIV_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_DIV_H_

#include "structure/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class DivConstScalarKernel : public SingleInputWithoutBufferKernel {
public:
  DivConstScalarKernel(Type type, float64_t constant);
  DivConstScalarKernel(const DivConstScalarKernel &div_kernel) = delete;
  DivConstScalarKernel(DivConstScalarKernel &&div_kernel) = default;
  virtual ~DivConstScalarKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "DivConstScalarKernel";
  Type type_;
  float64_t constant_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif