#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_TANH_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_TANH_H_

#include "structure/kernel/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

class TanhKernel : public SingleInputWithoutBufferKernel {
public:
  TanhKernel() = default;
  TanhKernel(const TanhKernel &other) = delete;
  TanhKernel(TanhKernel &&other) = default;
  virtual ~TanhKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "TanhKernel";
};

} // namespace kernel
} // namespace cpu_transformers

#endif
