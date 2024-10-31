#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_RELU_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_RELU_H_

#include "structure/kernel/kernel/kernel.h"

namespace fluidml {
namespace kernel {

class ReluKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "ReluKernel";
  ReluKernel() = default;
  ReluKernel(const ReluKernel &) = delete;
  ReluKernel(ReluKernel &&) = default;
  virtual ~ReluKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace fluidml

#endif
