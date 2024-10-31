#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_RESHAPE_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_RESHAPE_H_

#include "structure/kernel/kernel/kernel.h"

namespace fluidml {
namespace kernel {

class ReshapeKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "ReshapeKernel";
  ReshapeKernel() = default;
  ReshapeKernel(const ReshapeKernel &) = delete;
  ReshapeKernel(ReshapeKernel &&) = default;
  virtual ~ReshapeKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace fluidml

#endif
