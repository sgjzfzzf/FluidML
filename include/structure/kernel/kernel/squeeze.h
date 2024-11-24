#ifndef FLUIMDL_STRUCTURE_KERNEL_KERNEL_SQUEEZE_H_
#define FLUIMDL_STRUCTURE_KERNEL_KERNEL_SQUEEZE_H_

#include "structure/kernel/kernel/kernel.h"

namespace fluidml {
namespace kernel {

class SqueezeKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "SqueezeKernel";
  SqueezeKernel(std::vector<int64_t> &&axes);
  SqueezeKernel(const SqueezeKernel &) = delete;
  SqueezeKernel(SqueezeKernel &&) = default;
  virtual ~SqueezeKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const std::vector<int64_t> axes_;
};

} // namespace kernel
} // namespace fluidml

#endif
