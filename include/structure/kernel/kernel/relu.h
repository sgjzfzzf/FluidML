#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_RELU_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_RELU_H_

#include "structure/kernel/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

class ReluKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "ReluKernel";
  ReluKernel() = default;
  ReluKernel(const ReluKernel &) = delete;
  ReluKernel(ReluKernel &&) = default;
  virtual ~ReluKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
