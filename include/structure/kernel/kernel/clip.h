#ifndef FLUIDML_KERNEL_KERNEL_CLIP_H_
#define FLUIDML_KERNEL_KERNEL_CLIP_H_

#include "structure/kernel/kernel/kernel.h"
#include "utils/float.h"

namespace fluidml {
namespace kernel {

class ClipKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "ClipKernel";
  ClipKernel(float32_t min, float32_t max);
  ClipKernel(const ClipKernel &) = delete;
  ClipKernel(ClipKernel &&) = default;
  virtual ~ClipKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const float32_t min_;
  const float32_t max_;
};

} // namespace kernel
} // namespace fluidml

#endif
