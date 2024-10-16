#include "structure/kernel/generator/sub.h"

namespace cpu_transformers {
namespace kernel {

class SubConstantLhsKernelGeneratorImpl : public SubConstantLhsKernelGenerator {
public:
  SubConstantLhsKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                                    Type type, float64_t value);
  SubConstantLhsKernelGeneratorImpl(
      const SubConstantLhsKernelGeneratorImpl &generator) = delete;
  SubConstantLhsKernelGeneratorImpl(
      SubConstantLhsKernelGeneratorImpl &&generator) = default;
  virtual ~SubConstantLhsKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<SubConstantLhsKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Type type_;
  const float64_t value_;
};

std::unique_ptr<SubConstantLhsKernelGenerator>
SubConstantLhsKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                                    Type type, float64_t value) {
  return std::make_unique<SubConstantLhsKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), type, value);
}

SubConstantLhsKernelGeneratorImpl::SubConstantLhsKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, Type type, float64_t value)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      type_(type), value_(value) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
SubConstantLhsKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<SubConstantLhsKernel>
SubConstantLhsKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                         llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<SubConstantLhsKernel>(type_, value_);
}

const Meta &SubConstantLhsKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &SubConstantLhsKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string SubConstantLhsKernelGeneratorImpl::GetKernelName() const {
  return SubConstantLhsKernel::kKernelName;
}

} // namespace kernel
} // namespace cpu_transformers
