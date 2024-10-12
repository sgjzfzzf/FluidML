#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_RESHAPE_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_RESHAPE_H_

#include "structure/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

class ReshapeKernel : public SingleInputWithoutBufferKernel {
public:
  ReshapeKernel() = default;
  ReshapeKernel(const ReshapeKernel &) = delete;
  ReshapeKernel(ReshapeKernel &&) = default;
  virtual ~ReshapeKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "ReshapeKernel";
};

} // namespace kernel
} // namespace cpu_transformers

#endif
