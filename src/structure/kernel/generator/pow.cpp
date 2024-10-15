#include "structure/kernel/generator/pow.h"

namespace cpu_transformers {
namespace kernel {

class PowKernelGeneratorImpl : public PowKernelGenerator {
public:
  PowKernelGeneratorImpl(Type type, float64_t exp);
  PowKernelGeneratorImpl(const PowKernelGeneratorImpl &generator) = delete;
  PowKernelGeneratorImpl(PowKernelGeneratorImpl &&generator) = default;
  virtual ~PowKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<PowKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const Type type_;
  const float64_t exp_;
};

std::unique_ptr<PowKernelGenerator> PowKernelGenerator::Make(Type type,
                                                             float64_t exp) {
  return std::make_unique<PowKernelGeneratorImpl>(type, exp);
}

PowKernelGeneratorImpl::PowKernelGeneratorImpl(Type type, float64_t exp)
    : type_(type), exp_(exp) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
PowKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<PowKernel>
PowKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                              llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<PowKernel>(type_, exp_);
}

} // namespace kernel
} // namespace cpu_transformers
