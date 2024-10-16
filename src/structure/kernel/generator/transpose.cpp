#include "structure/kernel/generator/transpose.h"

namespace cpu_transformers {
namespace kernel {

class TransposeKernelGeneratorImpl : public TransposeKernelGenerator {
public:
  TransposeKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                               std::vector<int64_t> &&perms);
  TransposeKernelGeneratorImpl(const TransposeKernelGeneratorImpl &generator) =
      delete;
  TransposeKernelGeneratorImpl(TransposeKernelGeneratorImpl &&generator) =
      default;
  virtual ~TransposeKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<TransposeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const std::vector<int64_t> perms_;
};

std::unique_ptr<TransposeKernelGenerator>
TransposeKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                               std::vector<int64_t> &&perms) {
  return std::make_unique<TransposeKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(perms));
}

TransposeKernelGeneratorImpl::TransposeKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&perms)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      perms_(std::move(perms)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
TransposeKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<TransposeKernel>
TransposeKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                    llvm::ArrayRef<size_t> output_layout) {
  std::vector<int64_t> perms = perms_;
  return std::make_shared<TransposeKernel>(std::move(perms));
}

const Meta &TransposeKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &TransposeKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string TransposeKernelGeneratorImpl::GetKernelName() const {
  return TransposeKernel::kKernelName;
}

} // namespace kernel
} // namespace cpu_transformers