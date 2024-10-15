#include "structure/kernel/generator/sub.h"

namespace cpu_transformers {
namespace kernel {

class SubConstantLhsKernelGeneratorImpl : public SubConstantLhsKernelGenerator {
public:
  SubConstantLhsKernelGeneratorImpl(Type type, float64_t value);
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

private:
  const Type type_;
  const float64_t value_;
};

std::unique_ptr<SubConstantLhsKernelGenerator>
SubConstantLhsKernelGenerator::Make(Type type, float64_t value) {
  return std::make_unique<SubConstantLhsKernelGeneratorImpl>(type, value);
}

SubConstantLhsKernelGeneratorImpl::SubConstantLhsKernelGeneratorImpl(
    Type type, float64_t value)
    : type_(type), value_(value) {}

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

} // namespace kernel
} // namespace cpu_transformers
