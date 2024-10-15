#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_ERF_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_ERF_H_

#include "structure/kernel/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

class ErfKernel : public SingleInputWithoutBufferKernel {
public:
  ErfKernel() = default;
  ErfKernel(const ErfKernel &erf_kernel) = delete;
  ErfKernel(ErfKernel &&erf_kernel) = default;
  virtual ~ErfKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "ErfKernel";
};

} // namespace kernel
} // namespace cpu_transformers

#endif