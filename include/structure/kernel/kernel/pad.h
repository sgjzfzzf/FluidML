#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_PAD_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_PAD_H_

#include "structure/kernel/kernel/kernel.h"

namespace fluidml {
namespace kernel {

class PadKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "PadKernel";
  PadKernel(std::vector<std::tuple<int64_t, int64_t>> &&pads);
  PadKernel(const PadKernel &) = delete;
  PadKernel(PadKernel &&) = default;
  virtual ~PadKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const std::vector<std::tuple<int64_t, int64_t>> pads_;
};

} // namespace kernel
} // namespace fluidml

#endif
