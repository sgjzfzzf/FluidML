#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_FLATTEN_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_FLATTEN_H_

#include "structure/kernel/kernel/kernel.h"

namespace fluidml {
namespace kernel {

class FlattenKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "FlattenKernel";
  FlattenKernel(int64_t axis);
  FlattenKernel(const FlattenKernel &) = delete;
  FlattenKernel(FlattenKernel &&) = default;
  virtual ~FlattenKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const int64_t axis_;
};

} // namespace kernel
} // namespace fluidml

#endif
