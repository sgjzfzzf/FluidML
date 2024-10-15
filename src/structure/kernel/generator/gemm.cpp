#include "structure/kernel/generator/gemm.h"

namespace cpu_transformers {
namespace kernel {

class GemmConstantWeightsBiasKernelGeneratorImpl
    : public GemmConstantWeightsBiasKernelGenerator {
public:
  GemmConstantWeightsBiasKernelGeneratorImpl(float64_t alpha, float64_t beta,
                                             bool transA, bool transB,
                                             Tensor &&weights, Tensor &&bias);
  GemmConstantWeightsBiasKernelGeneratorImpl(
      const GemmConstantWeightsBiasKernelGeneratorImpl &generator) = delete;
  GemmConstantWeightsBiasKernelGeneratorImpl(
      GemmConstantWeightsBiasKernelGeneratorImpl &&generator) = default;
  virtual ~GemmConstantWeightsBiasKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<GemmConstantWeightsBiasKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const float64_t alpha_;
  const float64_t beta_;
  const bool transA_;
  const bool transB_;
  const Tensor weights_;
  const Tensor bias_;
};

std::unique_ptr<GemmConstantWeightsBiasKernelGenerator>
GemmConstantWeightsBiasKernelGenerator::Make(float64_t alpha, float64_t beta,
                                             bool transA, bool transB,
                                             Tensor &&weights, Tensor &&bias) {
  return std::make_unique<GemmConstantWeightsBiasKernelGeneratorImpl>(
      alpha, beta, transA, transB, std::move(weights), std::move(bias));
}

GemmConstantWeightsBiasKernelGeneratorImpl::
    GemmConstantWeightsBiasKernelGeneratorImpl(float64_t alpha, float64_t beta,
                                               bool transA, bool transB,
                                               Tensor &&weights, Tensor &&bias)
    : alpha_(alpha), beta_(beta), transA_(transA), transB_(transB),
      weights_(weights), bias_(bias) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
GemmConstantWeightsBiasKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<GemmConstantWeightsBiasKernel>
GemmConstantWeightsBiasKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  Tensor weights = weights_, bias = bias_;
  return std::make_shared<GemmConstantWeightsBiasKernel>(
      alpha_, beta_, transA_, transB_, std::move(weights), std::move(bias));
}

} // namespace kernel
} // namespace cpu_transformers
