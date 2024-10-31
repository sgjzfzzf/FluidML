#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_CAST_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_CAST_H_

#include "structure/kernel/kernel/kernel.h"

namespace fluidml {
namespace kernel {

class CastKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "CastKernel";
  CastKernel() = default;
  CastKernel(const CastKernel &) = delete;
  CastKernel(CastKernel &&) = default;
  virtual ~CastKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace fluidml

#endif
