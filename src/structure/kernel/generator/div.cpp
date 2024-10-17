#include "structure/kernel/generator/div.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class DivConstantRhsKernelGeneratorImpl : public DivConstantRhsKernelGenerator {
public:
  DivConstantRhsKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                                    Type type, float64_t constant);
  DivConstantRhsKernelGeneratorImpl(
      const DivConstantRhsKernelGeneratorImpl &generator) = delete;
  DivConstantRhsKernelGeneratorImpl(
      DivConstantRhsKernelGeneratorImpl &&generator) = default;
  virtual ~DivConstantRhsKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<DivConstantRhsKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const DivConstantRhsKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Type type_;
  const float64_t constant_;
};

std::unique_ptr<SingleInputWithoutBufferKernelGenerator>
DivConstantRhsKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                                    Type type, float64_t constant) {
  return std::make_unique<DivConstantRhsKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), type, constant);
}

DivConstantRhsKernelGeneratorImpl::DivConstantRhsKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, Type type, float64_t constant)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      type_(type), constant_(constant) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
DivConstantRhsKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<DivConstantRhsKernel>
DivConstantRhsKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                         llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<DivConstantRhsKernel>(type_, constant_);
}

const Meta &DivConstantRhsKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &DivConstantRhsKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string DivConstantRhsKernelGeneratorImpl::GetKernelName() const {
  return DivConstantRhsKernel::kKernelName;
}

size_t DivConstantRhsKernelGeneratorImpl::GetHashCode() const {
  size_t hash  = typeid(DivConstantRhsKernelGeneratorImpl).hash_code();
  hash ^= input_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= output_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= std::hash<Type>()(type_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^=
      std::hash<float64_t>()(constant_) + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool DivConstantRhsKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const DivConstantRhsKernelGeneratorImpl *other_ptr =
          dynamic_cast<const DivConstantRhsKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool DivConstantRhsKernelGeneratorImpl::Equals(
    const DivConstantRhsKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ &&
         output_meta_ == other.output_meta_ && type_ == other.type_ &&
         constant_ == other.constant_;
}

} // namespace kernel
} // namespace cpu_transformers
