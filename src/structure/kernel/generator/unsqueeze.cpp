#include "structure/kernel/generator/unsqueeze.h"

namespace cpu_transformers {
namespace kernel {

class UnSqueezeKernelGeneratorImpl : public UnSqueezeKernelGenerator {
public:
  UnSqueezeKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                               std::vector<int64_t> &&axes);
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
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const std::vector<int64_t> axes_;
};

std::unique_ptr<UnSqueezeKernelGenerator>
UnSqueezeKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                               std::vector<int64_t> &&axes) {
  return std::make_unique<UnSqueezeKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(axes));
}

UnSqueezeKernelGeneratorImpl::UnSqueezeKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&axes)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      axes_(std::move(axes)) {}

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

const Meta &UnSqueezeKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &UnSqueezeKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string UnSqueezeKernelGeneratorImpl::GetKernelName() const {
  return UnSqueezeKernel::kKernelName;
}

} // namespace kernel
} // namespace cpu_transformers