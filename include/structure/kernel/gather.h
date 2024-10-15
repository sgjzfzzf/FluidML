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
  const int64_t axis_;
  const int64_t index_;
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
  const Tensor data_;
};

class GatherConstantIndexScalarKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~GatherConstantIndexScalarKernelGenerator() override = default;
  virtual std::shared_ptr<GatherConstantIndexScalarKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<GatherConstantIndexScalarKernelGenerator>
  Make(int64_t axis, int64_t index);

protected:
  GatherConstantIndexScalarKernelGenerator() = default;
  GatherConstantIndexScalarKernelGenerator(
      const GatherConstantIndexScalarKernelGenerator &generator) = delete;
  GatherConstantIndexScalarKernelGenerator(
      GatherConstantIndexScalarKernelGenerator &&generator) = default;
};

class GatherConstantDataTensorKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~GatherConstantDataTensorKernelGenerator() override = default;
  virtual std::shared_ptr<GatherConstantDataTensorKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<GatherConstantDataTensorKernelGenerator>
  Make(Tensor &&data);

protected:
  GatherConstantDataTensorKernelGenerator() = default;
  GatherConstantDataTensorKernelGenerator(
      const GatherConstantDataTensorKernelGenerator &generator) = delete;
  GatherConstantDataTensorKernelGenerator(
      GatherConstantDataTensorKernelGenerator &&generator) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
