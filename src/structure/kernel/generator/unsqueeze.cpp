#include "structure/kernel/generator/unsqueeze.h"

namespace cpu_transformers {
namespace kernel {

class UnSqueezeKernelGeneratorImpl : public UnSqueezeKernelGenerator {
public:
  UnSqueezeKernelGeneratorImpl(std::vector<int64_t> axes);
  UnSqueezeKernelGeneratorImpl(const UnSqueezeKernelGeneratorImpl &generator) =
      delete;
  UnSqueezeKernelGeneratorImpl(UnSqueezeKernelGeneratorImpl &&generator) =
      default;
  virtual ~UnSqueezeKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<UnSqueezeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const std::vector<int64_t> axes_;
};

std::unique_ptr<UnSqueezeKernelGenerator>
UnSqueezeKernelGenerator::Make(std::vector<int64_t> axes) {
  return std::make_unique<UnSqueezeKernelGeneratorImpl>(std::move(axes));
}

UnSqueezeKernelGeneratorImpl::UnSqueezeKernelGeneratorImpl(
    std::vector<int64_t> axes)
    : axes_(std::move(axes)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
UnSqueezeKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<UnSqueezeKernel>
UnSqueezeKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                    llvm::ArrayRef<size_t> output_layout) {
  std::vector<int64_t> axes = axes_;
  return std::make_shared<UnSqueezeKernel>(std::move(axes));
}

} // namespace kernel
} // namespace cpu_transformers