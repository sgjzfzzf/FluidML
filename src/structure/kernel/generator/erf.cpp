#include "structure/kernel/generator/erf.h"

namespace cpu_transformers {
namespace kernel {

class ErfKernelGeneratorImpl : public ErfKernelGenerator {
public:
  ErfKernelGeneratorImpl() = default;
  ErfKernelGeneratorImpl(const ErfKernelGeneratorImpl &generator) = delete;
  ErfKernelGeneratorImpl(ErfKernelGeneratorImpl &&generator) = default;
  virtual ~ErfKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<ErfKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
};

std::unique_ptr<ErfKernelGenerator> ErfKernelGenerator::Make() {
  return std::make_unique<ErfKernelGeneratorImpl>();
}

std::shared_ptr<SingleInputWithoutBufferKernel>
ErfKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<ErfKernel>
ErfKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                              llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<ErfKernel>();
}

} // namespace kernel
} // namespace cpu_transformers
