#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_UNSQUEEZE_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_UNSQUEEZE_H_

#include "structure/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {
class UnSqueezeKernel : public Kernel {
public:
  UnSqueezeKernel(std::vector<int64_t> &&axes);
  UnSqueezeKernel(const UnSqueezeKernel &other) = delete;
  UnSqueezeKernel(UnSqueezeKernel &&other) = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &input, mlir::Value &output);

private:
  std::vector<int64_t> axes_;
};
} // namespace kernel
} // namespace cpu_transformers

#endif
