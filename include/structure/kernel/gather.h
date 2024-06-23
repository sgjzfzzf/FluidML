#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GATHER_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GATHER_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace cpu_transformers {
namespace kernel {

class GatherConstantIndexScalarKernel : public Kernel {
public:
  GatherConstantIndexScalarKernel(int64_t axis);
  GatherConstantIndexScalarKernel(
      const GatherConstantIndexScalarKernel &gather_kernel) = delete;
  GatherConstantIndexScalarKernel(
      GatherConstantIndexScalarKernel &&gather_kernel) = default;
  ~GatherConstantIndexScalarKernel() = default;
  void Run(mlir::OpBuilder &builder, mlir::Value &data, int64_t index,
           mlir::Value &output);

private:
  int64_t axis_;
};

class GatherConstantDataTensorKernel : public Kernel {
public:
  GatherConstantDataTensorKernel() = default;
  GatherConstantDataTensorKernel(
      const GatherConstantDataTensorKernel &gather_kernel) = delete;
  GatherConstantDataTensorKernel(
      GatherConstantDataTensorKernel &&gather_kernel) = default;
  ~GatherConstantDataTensorKernel() = default;
  void Run(mlir::OpBuilder &builder, const Tensor &data, mlir::Value &indices,
           mlir::Value &output);
};

} // namespace kernel
} // namespace cpu_transformers

#endif
