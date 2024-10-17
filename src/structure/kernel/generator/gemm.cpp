#include "structure/kernel/generator/gemm.h"
#include "utils/float.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class GemmConstantWeightsBiasKernelGeneratorImpl
    : public GemmConstantWeightsBiasKernelGenerator {
public:
  GemmConstantWeightsBiasKernelGeneratorImpl(Meta &&input_meta,
                                             Meta &&output_meta,
                                             float64_t alpha, float64_t beta,
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
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const GemmConstantWeightsBiasKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const float64_t alpha_;
  const float64_t beta_;
  const bool transA_;
  const bool transB_;
  const Tensor weights_;
  const Tensor bias_;
};

std::unique_ptr<GemmConstantWeightsBiasKernelGenerator>
GemmConstantWeightsBiasKernelGenerator::Make(Meta &&input_meta,
                                             Meta &&output_meta,
                                             float64_t alpha, float64_t beta,
                                             bool transA, bool transB,
                                             Tensor &&weights, Tensor &&bias) {
  return std::make_unique<GemmConstantWeightsBiasKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), alpha, beta, transA,
      transB, std::move(weights), std::move(bias));
}

GemmConstantWeightsBiasKernelGeneratorImpl::
    GemmConstantWeightsBiasKernelGeneratorImpl(Meta &&input_meta,
                                               Meta &&output_meta,
                                               float64_t alpha, float64_t beta,
                                               bool transA, bool transB,
                                               Tensor &&weights, Tensor &&bias)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      alpha_(alpha), beta_(beta), transA_(transA), transB_(transB),
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

const Meta &GemmConstantWeightsBiasKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &GemmConstantWeightsBiasKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string GemmConstantWeightsBiasKernelGeneratorImpl::GetKernelName() const {
  return GemmConstantWeightsBiasKernel::kKernelName;
}

size_t GemmConstantWeightsBiasKernelGeneratorImpl::GetHashCode() const {
  std::hash<float64_t> f64_hash;
  std::hash<bool> bool_hash;
  size_t hash = typeid(GemmConstantWeightsBiasKernelGeneratorImpl).hash_code();
  hash ^= input_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= output_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= f64_hash(alpha_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= f64_hash(beta_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= bool_hash(transA_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= bool_hash(transB_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= weights_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= bias_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool GemmConstantWeightsBiasKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const GemmConstantWeightsBiasKernelGeneratorImpl *other_ptr =
          dynamic_cast<const GemmConstantWeightsBiasKernelGeneratorImpl *>(
              &other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool GemmConstantWeightsBiasKernelGeneratorImpl::Equals(
    const GemmConstantWeightsBiasKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ &&
         output_meta_ == other.output_meta_ && alpha_ == other.alpha_ &&
         beta_ == other.beta_ && transA_ == other.transA_ &&
         transB_ == other.transB_ && weights_ == other.weights_ &&
         bias_ == other.bias_;
}

} // namespace kernel
} // namespace cpu_transformers
