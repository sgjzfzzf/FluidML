#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_TANH_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_TANH_H_

#include "structure/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {
class TanhKernel : public Kernel {
public:
  TanhKernel() = default;
  TanhKernel(const TanhKernel &other) = delete;
  TanhKernel(TanhKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &input, mlir::Value &output);
};
} // namespace kernel
} // namespace cpu_transformers

#endif
