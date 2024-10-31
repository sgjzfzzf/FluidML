#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_ERF_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_ERF_H_

#include "structure/kernel/kernel/kernel.h"

namespace fluidml {
namespace kernel {

class ErfKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "ErfKernel";
  ErfKernel() = default;
  ErfKernel(const ErfKernel &) = delete;
  ErfKernel(ErfKernel &&) = default;
  virtual ~ErfKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace fluidml

#endif