#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_NOT_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_NOT_H_

#include "structure/kernel/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

class NotKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "NotKernel";
  NotKernel() = default;
  NotKernel(const NotKernel &) = delete;
  NotKernel(NotKernel &&) = default;
  virtual ~NotKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
