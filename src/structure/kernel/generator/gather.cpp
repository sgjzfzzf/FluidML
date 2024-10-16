#include "structure/kernel/generator/gather.h"

namespace cpu_transformers {
namespace kernel {

class GatherConstantIndexScalarKernelGeneratorImpl
    : public GatherConstantIndexScalarKernelGenerator {
public:
  GatherConstantIndexScalarKernelGeneratorImpl(Meta &&input_meta,
                                               Meta &&output_meta, int64_t axis,
                                               int64_t index);
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
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const int64_t axis_;
  const int64_t index_;
};

class GatherConstantDataTensorKernelGeneratorImpl
    : public GatherConstantDataTensorKernelGenerator {
public:
  GatherConstantDataTensorKernelGeneratorImpl(Meta &&input_meta,
                                              Meta &&output_meta,
                                              Tensor &&data);
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
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Tensor data_;
};

std::unique_ptr<GatherConstantIndexScalarKernelGenerator>
GatherConstantIndexScalarKernelGenerator::Make(Meta &&input_meta,
                                               Meta &&output_meta, int64_t axis,
                                               int64_t index) {
  return std::make_unique<GatherConstantIndexScalarKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), axis, index);
}

std::unique_ptr<GatherConstantDataTensorKernelGenerator>
GatherConstantDataTensorKernelGenerator::Make(Meta &&input_meta,
                                              Meta &&output_meta,
                                              Tensor &&data) {
  return std::make_unique<GatherConstantDataTensorKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(data));
}

GatherConstantIndexScalarKernelGeneratorImpl::
    GatherConstantIndexScalarKernelGeneratorImpl(Meta &&input_meta,
                                                 Meta &&output_meta,
                                                 int64_t axis, int64_t index)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      axis_(axis), index_(index) {}

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

const Meta &GatherConstantIndexScalarKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &
GatherConstantIndexScalarKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string
GatherConstantIndexScalarKernelGeneratorImpl::GetKernelName() const {
  return GatherConstantIndexScalarKernel::kKernelName;
}

GatherConstantDataTensorKernelGeneratorImpl::
    GatherConstantDataTensorKernelGeneratorImpl(Meta &&input_meta,
                                                Meta &&output_meta,
                                                Tensor &&data)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      data_(std::move(data)) {}

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

const Meta &GatherConstantDataTensorKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &GatherConstantDataTensorKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string GatherConstantDataTensorKernelGeneratorImpl::GetKernelName() const {
  return GatherConstantDataTensorKernel::kKernelName;
}

} // namespace kernel
} // namespace cpu_transformers
