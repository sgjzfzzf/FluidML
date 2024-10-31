#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_SQRT_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_SQRT_H_

#include "structure/kernel/kernel/kernel.h"

namespace fluidml {
namespace kernel {

class SqrtKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "SqrtKernel";
  SqrtKernel() = default;
  SqrtKernel(const SqrtKernel &) = delete;
  SqrtKernel(SqrtKernel &&) = default;
  virtual ~SqrtKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace fluidml

#endif
