#include "structure/kernel/generator/unsqueeze_sub_mul.h"

namespace cpu_transformers {
namespace kernel {

class UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl
    : public UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator {
public:
  UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl(
      std::vector<int64_t> &&unsqueeze_axes, const Type &sub_type,
      float64_t sub_val, const Type &mul_type, float64_t mul_val);
  UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl(
      const UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl &generator) =
      delete;
  UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl(
      UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl &&generator) =
      default;
  virtual ~UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<UnsqueezeSubLhsScalarMulRhsScalarKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const std::vector<int64_t> unsqueeze_axes_;
  const Type sub_type_;
  const float64_t sub_val_;
  const Type mul_type_;
  const float64_t mul_val_;
};

std::unique_ptr<UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator>
UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator::Make(
    std::vector<int64_t> &&unsqueeze_axes, const Type &sub_type,
    float64_t sub_val, const Type &mul_type, float64_t mul_val) {
  return std::make_unique<UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl>(
      std::move(unsqueeze_axes), sub_type, sub_val, mul_type, mul_val);
}

UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl::
    UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl(
        std::vector<int64_t> &&unsqueeze_axes, const Type &sub_type,
        float64_t sub_val, const Type &mul_type, float64_t mul_val)
    : unsqueeze_axes_(std::move(unsqueeze_axes)), sub_type_(sub_type),
      sub_val_(sub_val), mul_type_(mul_type), mul_val_(mul_val) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl::
    YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                        llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<UnsqueezeSubLhsScalarMulRhsScalarKernel>
UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  std::vector<int64_t> unsqueeze_axes = unsqueeze_axes_;
  return std::make_shared<UnsqueezeSubLhsScalarMulRhsScalarKernel>(
      std::move(unsqueeze_axes), sub_type_, sub_val_, mul_type_, mul_val_);
}

} // namespace kernel
} // namespace cpu_transformers
