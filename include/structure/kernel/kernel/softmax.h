#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_SOFTMAX_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_SOFTMAX_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "structure/kernel/kernel/kernel.h"
#include <cstdint>

namespace fluidml {
namespace kernel {

class SoftmaxKernel : public SingleInputWithBufferKernel {
public:
  static constexpr char kKernelName[] = "SoftmaxKernel";
  SoftmaxKernel(int64_t axis);
  SoftmaxKernel(const SoftmaxKernel &) = delete;
  SoftmaxKernel(SoftmaxKernel &&) = default;
  virtual ~SoftmaxKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input, mlir::Value &output,
           mlir::Value &buffer) const override;

private:
  const int64_t axis_;
};

} // namespace kernel
} // namespace fluidml

#endif
