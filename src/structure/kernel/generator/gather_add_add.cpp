#include "structure/kernel/kernel/gather_add_add.h"
#include "structure/kernel/generator/gather_add_add.h"

namespace cpu_transformers {
namespace kernel {

class GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGeneratorImpl
    : public GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator {
public:
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGeneratorImpl(
      Tensor &&data, Tensor &&add0_weight, Tensor &&add1_weight);
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGeneratorImpl(
      const GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGeneratorImpl
          &generator) = delete;
  GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGeneratorImpl(
      GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGeneratorImpl
          &&generator) = default;
  virtual ~GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGeneratorImpl() =
      default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const Tensor data_;
  const Tensor add0_weight_;
  const Tensor add1_weight_;
};

std::unique_ptr<GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator>
GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator::Make(
    Tensor &&data, Tensor &&add0_weight, Tensor &&add1_weight) {
  return std::make_unique<
      GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGeneratorImpl>(
      std::move(data), std::move(add0_weight), std::move(add1_weight));
}

GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGeneratorImpl::
    GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGeneratorImpl(
        Tensor &&data, Tensor &&add0_weight, Tensor &&add1_weight)
    : data_(std::move(data)), add0_weight_(std::move(add0_weight)),
      add1_weight_(std::move(add1_weight)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGeneratorImpl::
    YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                        llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel>
GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  Tensor data = data_, add0_weight = add0_weight_, add1_weight = add1_weight_;
  return std::make_shared<
      GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel>(
      std::move(data), std::move(add0_weight), std::move(add1_weight));
}

} // namespace kernel
} // namespace cpu_transformers
