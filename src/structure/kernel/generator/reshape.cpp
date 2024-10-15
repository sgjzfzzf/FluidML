#include "structure/kernel/generator/reshape.h"

namespace cpu_transformers {
namespace kernel {

class ReshapeKernelGeneratorImpl : public ReshapeKernelGenerator {
public:
  ReshapeKernelGeneratorImpl() = default;
  ReshapeKernelGeneratorImpl(const ReshapeKernelGeneratorImpl &generator) =
      delete;
  ReshapeKernelGeneratorImpl(ReshapeKernelGeneratorImpl &&generator) = default;
  virtual ~ReshapeKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<ReshapeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
};

std::unique_ptr<ReshapeKernelGenerator> ReshapeKernelGenerator::Make() {
  return std::make_unique<ReshapeKernelGeneratorImpl>();
}

std::shared_ptr<SingleInputWithoutBufferKernel>
ReshapeKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<ReshapeKernel>
ReshapeKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                  llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<ReshapeKernel>();
}

} // namespace kernel
} // namespace cpu_transformers
