#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_DROPOUT_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_DROPOUT_H_

#include "structure/kernel/kernel/kernel.h"
#include "utils/float.h"

namespace fluidml {
namespace kernel {

class DropoutKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "DropoutKernel";
  DropoutKernel(float64_t ratio);
  DropoutKernel(const DropoutKernel &) = delete;
  DropoutKernel(DropoutKernel &&) = default;
  virtual ~DropoutKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const float64_t ratio_;
};

} // namespace kernel
} // namespace fluidml

#endif
