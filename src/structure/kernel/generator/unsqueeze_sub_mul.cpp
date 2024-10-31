#include "structure/kernel/generator/unsqueeze_sub_mul.h"
#include <cstdint>

namespace fluidml {
namespace kernel {

class UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl
    : public UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator {
public:
  UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl(
      Meta &&input_meta, Meta &&output_meta,
      std::vector<int64_t> &&unsqueeze_axes, const Type &sub_type,
      float64_t sub_val, const Type &mul_type, float64_t mul_val);
  UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl(
      const UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl &) = delete;
  UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl(
      UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl &&) = default;
  virtual ~UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<UnsqueezeSubLhsScalarMulRhsScalarKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(
      const UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const std::vector<int64_t> unsqueeze_axes_;
  const Type sub_type_;
  const float64_t sub_val_;
  const Type mul_type_;
  const float64_t mul_val_;
};

std::unique_ptr<UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator>
UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator::Make(
    Meta &&input_meta, Meta &&output_meta,
    std::vector<int64_t> &&unsqueeze_axes, const Type &sub_type,
    float64_t sub_val, const Type &mul_type, float64_t mul_val) {
  return std::make_unique<UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(unsqueeze_axes),
      sub_type, sub_val, mul_type, mul_val);
}

UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl::
    UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl(
        Meta &&input_meta, Meta &&output_meta,
        std::vector<int64_t> &&unsqueeze_axes, const Type &sub_type,
        float64_t sub_val, const Type &mul_type, float64_t mul_val)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      unsqueeze_axes_(std::move(unsqueeze_axes)), sub_type_(sub_type),
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

const Meta &
UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &
UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string
UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl::GetKernelName() const {
  return UnsqueezeSubLhsScalarMulRhsScalarKernel::kKernelName;
}

size_t
UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl::GetHashCode() const {
  std::hash<Type> type_hash;
  std::hash<int64_t> i64_hash;
  std::hash<float64_t> f64_hash;
  size_t hash =
      typeid(UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode();
  hash ^= GetOutputMeta().GetHashCode();
  for (int64_t unsqueeze_axis : unsqueeze_axes_) {
    hash ^= i64_hash(unsqueeze_axis);
  }
  hash ^= type_hash(sub_type_);
  hash ^= f64_hash(sub_val_);
  hash ^= type_hash(mul_type_);
  hash ^= f64_hash(mul_val_);
  return hash;
}

bool UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl *other_ptr =
          dynamic_cast<
              const UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl *>(
              &other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl::Equals(
    const UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ &&
         output_meta_ == other.output_meta_ &&
         unsqueeze_axes_ == other.unsqueeze_axes_ &&
         sub_type_ == other.sub_type_ && sub_val_ == other.sub_val_ &&
         mul_type_ == other.mul_type_ && mul_val_ == other.mul_val_;
}

} // namespace kernel
} // namespace fluidml
