#include "structure/kernel/generator/softmax.h"

namespace cpu_transformers {
namespace kernel {

class SoftmaxKernelGeneratorImpl : public SoftmaxKernelGenerator {
public:
  SoftmaxKernelGeneratorImpl(int64_t axis);
  SoftmaxKernelGeneratorImpl(const SoftmaxKernelGeneratorImpl &generator) =
      delete;
  SoftmaxKernelGeneratorImpl(SoftmaxKernelGeneratorImpl &&generator) = default;
  virtual ~SoftmaxKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithBufferKernel> YieldSingleInputWithBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<SoftmaxKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const int64_t axis_;
};

std::unique_ptr<SoftmaxKernelGenerator>
SoftmaxKernelGenerator::Make(int64_t axis) {
  return std::make_unique<SoftmaxKernelGeneratorImpl>(axis);
}

SoftmaxKernelGeneratorImpl::SoftmaxKernelGeneratorImpl(int64_t axis)
    : axis_(axis) {}

std::shared_ptr<SingleInputWithBufferKernel>
SoftmaxKernelGeneratorImpl::YieldSingleInputWithBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<SoftmaxKernel>
SoftmaxKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                  llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<SoftmaxKernel>(axis_);
}

} // namespace kernel
} // namespace cpu_transformers
