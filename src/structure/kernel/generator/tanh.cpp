#include "structure/kernel/generator/tanh.h"

namespace cpu_transformers {
namespace kernel {

class TanhKernelGeneratorImpl : public TanhKernelGenerator {
public:
  TanhKernelGeneratorImpl() = default;
  TanhKernelGeneratorImpl(const TanhKernelGeneratorImpl &generator) = delete;
  TanhKernelGeneratorImpl(TanhKernelGeneratorImpl &&generator) = default;
  virtual ~TanhKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<TanhKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  std::string GetKernelName() const override;
};

std::unique_ptr<TanhKernelGenerator> TanhKernelGenerator::Make() {
  return std::make_unique<TanhKernelGeneratorImpl>();
}

std::shared_ptr<SingleInputWithoutBufferKernel>
TanhKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<TanhKernel>
TanhKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                               llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<TanhKernel>();
}

std::string TanhKernelGeneratorImpl::GetKernelName() const {
  return TanhKernel::kKernelName;
}

} // namespace kernel
} // namespace cpu_transformers
