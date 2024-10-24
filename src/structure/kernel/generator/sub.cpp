#include "structure/kernel/generator/sub.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class SubConstantLhsKernelGeneratorImpl : public SubConstantLhsKernelGenerator {
public:
  SubConstantLhsKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                                    Type type, float64_t value);
  SubConstantLhsKernelGeneratorImpl(const SubConstantLhsKernelGeneratorImpl &) =
      delete;
  SubConstantLhsKernelGeneratorImpl(SubConstantLhsKernelGeneratorImpl &&) =
      default;
  virtual ~SubConstantLhsKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<SubConstantLhsKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const SubConstantLhsKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Type type_;
  const float64_t value_;
};

class SubCommonKernelGeneratorImpl : public SubCommonKernelGenerator {
public:
  SubCommonKernelGeneratorImpl(Meta &&lhs_meta, Meta &&rhs_meta,
                               Meta &&output_meta);
  SubCommonKernelGeneratorImpl(const SubCommonKernelGeneratorImpl &) = delete;
  SubCommonKernelGeneratorImpl(SubCommonKernelGeneratorImpl &&) = default;
  virtual ~SubCommonKernelGeneratorImpl() = default;
  std::shared_ptr<DoubleInputsWithoutBufferKernel>
  YieldDoubleInputsWithoutBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<SubCommonKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetLhsMeta() const override;
  const Meta &GetRhsMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const SubCommonKernelGeneratorImpl &other) const;

private:
  const Meta lhs_meta_;
  const Meta rhs_meta_;
  const Meta output_meta_;
};

std::unique_ptr<SubConstantLhsKernelGenerator>
SubConstantLhsKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                                    Type type, float64_t value) {
  return std::make_unique<SubConstantLhsKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), type, value);
}

std::unique_ptr<SubCommonKernelGenerator>
SubCommonKernelGenerator::Make(Meta &&lhs_meta, Meta &&rhs_meta,
                               Meta &&output_meta) {
  return std::make_unique<SubCommonKernelGeneratorImpl>(
      std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta));
}

SubConstantLhsKernelGeneratorImpl::SubConstantLhsKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, Type type, float64_t value)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      type_(type), value_(value) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
SubConstantLhsKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<SubConstantLhsKernel>
SubConstantLhsKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                         llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<SubConstantLhsKernel>(type_, value_);
}

const Meta &SubConstantLhsKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &SubConstantLhsKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string SubConstantLhsKernelGeneratorImpl::GetKernelName() const {
  return SubConstantLhsKernel::kKernelName;
}

size_t SubConstantLhsKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(SubConstantLhsKernelGeneratorImpl).hash_code();
  std::hash<Type> type_hash;
  std::hash<float64_t> f64_hash;
  hash ^= input_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= output_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= type_hash(type_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= f64_hash(value_) + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool SubConstantLhsKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const SubConstantLhsKernelGeneratorImpl *other_ptr =
          dynamic_cast<const SubConstantLhsKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool SubConstantLhsKernelGeneratorImpl::Equals(
    const SubConstantLhsKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ &&
         output_meta_ == other.output_meta_ && type_ == other.type_ &&
         value_ == other.value_;
}

SubCommonKernelGeneratorImpl::SubCommonKernelGeneratorImpl(Meta &&lhs_meta,
                                                           Meta &&rhs_meta,
                                                           Meta &&output_meta)
    : lhs_meta_(std::move(lhs_meta)), rhs_meta_(std::move(rhs_meta)),
      output_meta_(std::move(output_meta)) {}

std::shared_ptr<DoubleInputsWithoutBufferKernel>
SubCommonKernelGeneratorImpl::YieldDoubleInputsWithoutBufferKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return Yield(lhs_layout, rhs_layout, output_layout);
}

std::shared_ptr<SubCommonKernel>
SubCommonKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> lhs_layout,
                                    llvm::ArrayRef<size_t> rhs_layout,
                                    llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<SubCommonKernel>();
}

const Meta &SubCommonKernelGeneratorImpl::GetLhsMeta() const {
  return lhs_meta_;
}

const Meta &SubCommonKernelGeneratorImpl::GetRhsMeta() const {
  return rhs_meta_;
}

const Meta &SubCommonKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string SubCommonKernelGeneratorImpl::GetKernelName() const {
  return SubCommonKernel::kKernelName;
}

size_t SubCommonKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(SubCommonKernelGeneratorImpl).hash_code();
  hash ^= GetLhsMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetRhsMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool SubCommonKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const SubCommonKernelGeneratorImpl *other_ptr =
          dynamic_cast<const SubCommonKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool SubCommonKernelGeneratorImpl::Equals(
    const SubCommonKernelGeneratorImpl &other) const {
  return GetLhsMeta() == other.GetLhsMeta() &&
         GetRhsMeta() == other.GetRhsMeta() &&
         GetOutputMeta() == other.GetOutputMeta();
}

} // namespace kernel
} // namespace cpu_transformers
