#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_TANH_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_TANH_H_

#include "structure/kernel/kernel/kernel.h"

namespace fluidml {
namespace kernel {

class TanhKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "TanhKernel";
  TanhKernel() = default;
  TanhKernel(const TanhKernel &) = delete;
  TanhKernel(TanhKernel &&) = default;
  virtual ~TanhKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace fluidml

#endif
