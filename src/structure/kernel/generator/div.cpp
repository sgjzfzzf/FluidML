#include "structure/kernel/generator/div.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class DivConstantRhsKernelGeneratorImpl : public DivConstantRhsKernelGenerator {
public:
  DivConstantRhsKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                                    Type type, float64_t constant);
  DivConstantRhsKernelGeneratorImpl(const DivConstantRhsKernelGeneratorImpl &) =
      delete;
  DivConstantRhsKernelGeneratorImpl(DivConstantRhsKernelGeneratorImpl &&) =
      default;
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

class DivCommonKernelGeneratorImpl : public DivCommonKernelGenerator {
public:
  DivCommonKernelGeneratorImpl(Meta &&lhs_meta, Meta &&rhs_meta,
                               Meta &&output_meta);
  DivCommonKernelGeneratorImpl(const DivCommonKernelGeneratorImpl &) = delete;
  DivCommonKernelGeneratorImpl(DivCommonKernelGeneratorImpl &&) = default;
  virtual ~DivCommonKernelGeneratorImpl() = default;
  std::shared_ptr<DoubleInputsWithoutBufferKernel>
  YieldDoubleInputsWithoutBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<DivCommonKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetLhsMeta() const override;
  const Meta &GetRhsMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const DivCommonKernelGeneratorImpl &other) const;

private:
  const Meta lhs_meta_;
  const Meta rhs_meta_;
  const Meta output_meta_;
};

std::unique_ptr<SingleInputWithoutBufferKernelGenerator>
DivConstantRhsKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                                    Type type, float64_t constant) {
  return std::make_unique<DivConstantRhsKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), type, constant);
}

std::unique_ptr<DoubleInputsWithoutBufferKernelGenerator>
DivCommonKernelGenerator::Make(Meta &&lhs_meta, Meta &&rhs_meta,
                               Meta &&output_meta) {
  return std::make_unique<DivCommonKernelGeneratorImpl>(
      std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta));
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
  size_t hash = typeid(DivConstantRhsKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
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

DivCommonKernelGeneratorImpl::DivCommonKernelGeneratorImpl(Meta &&lhs_meta,
                                                           Meta &&rhs_meta,
                                                           Meta &&output_meta)
    : lhs_meta_(std::move(lhs_meta)), rhs_meta_(std::move(rhs_meta)),
      output_meta_(std::move(output_meta)) {}

std::shared_ptr<DoubleInputsWithoutBufferKernel>
DivCommonKernelGeneratorImpl::YieldDoubleInputsWithoutBufferKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return Yield(lhs_layout, rhs_layout, output_layout);
}

std::shared_ptr<DivCommonKernel>
DivCommonKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> lhs_layout,
                                    llvm::ArrayRef<size_t> rhs_layout,
                                    llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<DivCommonKernel>();
}

const Meta &DivCommonKernelGeneratorImpl::GetLhsMeta() const {
  return lhs_meta_;
}

const Meta &DivCommonKernelGeneratorImpl::GetRhsMeta() const {
  return rhs_meta_;
}

const Meta &DivCommonKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string DivCommonKernelGeneratorImpl::GetKernelName() const {
  return DivCommonKernel::kKernelName;
}

size_t DivCommonKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(DivCommonKernelGeneratorImpl).hash_code();
  hash ^= GetLhsMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetRhsMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool DivCommonKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const DivCommonKernelGeneratorImpl *other_ptr =
          dynamic_cast<const DivCommonKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool DivCommonKernelGeneratorImpl::Equals(
    const DivCommonKernelGeneratorImpl &other) const {
  return GetLhsMeta() == other.GetLhsMeta() &&
         GetRhsMeta() == other.GetRhsMeta() &&
         GetOutputMeta() == other.GetOutputMeta();
}

} // namespace kernel
} // namespace fluidml
