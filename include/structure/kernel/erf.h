#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_ERF_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_ERF_H_

#include "structure/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {
class ErfKernel : public Kernel {
public:
  ErfKernel() = default;
  ErfKernel(const ErfKernel &erf_kernel) = delete;
  ErfKernel(ErfKernel &&erf_kernel) = default;
  ~ErfKernel() = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &input, mlir::Value &output);
};
} // namespace kernel
} // namespace cpu_transformers

#endif