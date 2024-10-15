#include "structure/kernel/generator/transpose.h"

namespace cpu_transformers {
namespace kernel {

class TransposeKernelGeneratorImpl : public TransposeKernelGenerator {
public:
  TransposeKernelGeneratorImpl(std::vector<int64_t> &&perms);
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

private:
  const std::vector<int64_t> perms_;
};

std::unique_ptr<TransposeKernelGenerator>
TransposeKernelGenerator::Make(std::vector<int64_t> perms) {
  return std::make_unique<TransposeKernelGeneratorImpl>(std::move(perms));
}

TransposeKernelGeneratorImpl::TransposeKernelGeneratorImpl(
    std::vector<int64_t> &&perms)
    : perms_(std::move(perms)) {}

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

} // namespace kernel
} // namespace cpu_transformers