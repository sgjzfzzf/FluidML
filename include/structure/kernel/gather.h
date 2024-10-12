#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GATHER_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GATHER_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace cpu_transformers {
namespace kernel {

class GatherConstantIndexScalarKernel : public SingleInputWithoutBufferKernel {
public:
  GatherConstantIndexScalarKernel(int64_t axis, int64_t index);
  GatherConstantIndexScalarKernel(
      const GatherConstantIndexScalarKernel &gather_kernel) = delete;
  GatherConstantIndexScalarKernel(
      GatherConstantIndexScalarKernel &&gather_kernel) = default;
  virtual ~GatherConstantIndexScalarKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "GatherConstantIndexScalarKernel";
  int64_t axis_;
  int64_t index_;
};

class GatherConstantDataTensorKernel : public SingleInputWithoutBufferKernel {
public:
  GatherConstantDataTensorKernel(Tensor &&data);
  GatherConstantDataTensorKernel(
      const GatherConstantDataTensorKernel &gather_kernel) = delete;
  GatherConstantDataTensorKernel(
      GatherConstantDataTensorKernel &&gather_kernel) = default;
  virtual ~GatherConstantDataTensorKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "GatherConstantDataTensorKernel";
  Tensor data_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
