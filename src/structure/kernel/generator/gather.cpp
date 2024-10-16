#include "structure/kernel/generator/gather.h"

namespace cpu_transformers {
namespace kernel {

class GatherConstantIndexScalarKernelGeneratorImpl
    : public GatherConstantIndexScalarKernelGenerator {
public:
  GatherConstantIndexScalarKernelGeneratorImpl(int64_t axis, int64_t index);
  GatherConstantIndexScalarKernelGeneratorImpl(
      const GatherConstantIndexScalarKernelGeneratorImpl &generator) = delete;
  GatherConstantIndexScalarKernelGeneratorImpl(
      GatherConstantIndexScalarKernelGeneratorImpl &&generator) = default;
  virtual ~GatherConstantIndexScalarKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<GatherConstantIndexScalarKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  std::string GetKernelName() const override;

private:
  const int64_t axis_;
  const int64_t index_;
};

class GatherConstantDataTensorKernelGeneratorImpl
    : public GatherConstantDataTensorKernelGenerator {
public:
  GatherConstantDataTensorKernelGeneratorImpl(Tensor &&data);
  GatherConstantDataTensorKernelGeneratorImpl(
      const GatherConstantDataTensorKernelGeneratorImpl &generator) = delete;
  GatherConstantDataTensorKernelGeneratorImpl(
      GatherConstantDataTensorKernelGeneratorImpl &&generator) = default;
  virtual ~GatherConstantDataTensorKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<GatherConstantDataTensorKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  std::string GetKernelName() const override;

private:
  const Tensor data_;
};

std::unique_ptr<GatherConstantIndexScalarKernelGenerator>
GatherConstantIndexScalarKernelGenerator::Make(int64_t axis, int64_t index) {
  return std::make_unique<GatherConstantIndexScalarKernelGeneratorImpl>(axis,
                                                                        index);
}

std::unique_ptr<GatherConstantDataTensorKernelGenerator>
GatherConstantDataTensorKernelGenerator::Make(Tensor &&data) {
  return std::make_unique<GatherConstantDataTensorKernelGeneratorImpl>(
      std::move(data));
}

GatherConstantIndexScalarKernelGeneratorImpl::
    GatherConstantIndexScalarKernelGeneratorImpl(int64_t axis, int64_t index)
    : axis_(axis), index_(index) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
GatherConstantIndexScalarKernelGeneratorImpl::
    YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                        llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<GatherConstantIndexScalarKernel>
GatherConstantIndexScalarKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<GatherConstantIndexScalarKernel>(axis_, index_);
}

std::string
GatherConstantIndexScalarKernelGeneratorImpl::GetKernelName() const {
  return GatherConstantIndexScalarKernel::kKernelName;
}

GatherConstantDataTensorKernelGeneratorImpl::
    GatherConstantDataTensorKernelGeneratorImpl(Tensor &&data)
    : data_(std::move(data)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
GatherConstantDataTensorKernelGeneratorImpl::
    YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                        llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<GatherConstantDataTensorKernel>
GatherConstantDataTensorKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  Tensor data = data_;
  return std::make_shared<GatherConstantDataTensorKernel>(std::move(data));
}

std::string GatherConstantDataTensorKernelGeneratorImpl::GetKernelName() const {
  return GatherConstantDataTensorKernel::kKernelName;
}

} // namespace kernel
} // namespace cpu_transformers
