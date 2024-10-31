#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_UNSQUEEZE_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_UNSQUEEZE_H_

#include "structure/kernel/kernel/kernel.h"

namespace fluidml {
namespace kernel {

class UnsqueezeKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "UnsqueezeKernel";
  UnsqueezeKernel(std::vector<int64_t> &&axes);
  UnsqueezeKernel(const UnsqueezeKernel &) = delete;
  UnsqueezeKernel(UnsqueezeKernel &&) = default;
  virtual ~UnsqueezeKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const std::vector<int64_t> axes_;
};

} // namespace kernel
} // namespace fluidml

#endif
