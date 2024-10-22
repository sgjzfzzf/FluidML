#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_GATHER_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_GATHER_H_

#include "structure/kernel/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace cpu_transformers {
namespace kernel {

class GatherConstantIndexScalarKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "GatherConstantIndexScalarKernel";
  GatherConstantIndexScalarKernel(int64_t axis, int64_t index);
  GatherConstantIndexScalarKernel(const GatherConstantIndexScalarKernel &) =
      delete;
  GatherConstantIndexScalarKernel(GatherConstantIndexScalarKernel &&) = default;
  virtual ~GatherConstantIndexScalarKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const int64_t axis_;
  const int64_t index_;
};

class GatherConstantDataTensorKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "GatherConstantDataTensorKernel";
  GatherConstantDataTensorKernel(Tensor &&data);
  GatherConstantDataTensorKernel(const GatherConstantDataTensorKernel &) =
      delete;
  GatherConstantDataTensorKernel(GatherConstantDataTensorKernel &&) = default;
  virtual ~GatherConstantDataTensorKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const Tensor data_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
