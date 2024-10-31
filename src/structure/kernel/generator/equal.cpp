#include "structure/kernel/generator/equal.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class EqualKernelGeneratorImpl : public EqualKernelGenerator {
public:
  EqualKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta, Type type,
                           float64_t value);
  EqualKernelGeneratorImpl(const EqualKernelGeneratorImpl &) = delete;
  EqualKernelGeneratorImpl(EqualKernelGeneratorImpl &&) = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<EqualKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const EqualKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Type type_;
  const float64_t value_;
};

std::unique_ptr<EqualKernelGenerator>
EqualKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta, Type type,
                           float64_t value) {
  return std::make_unique<EqualKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), type, value);
}

EqualKernelGeneratorImpl::EqualKernelGeneratorImpl(Meta &&input_meta,
                                                   Meta &&output_meta,
                                                   Type type, float64_t value)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      type_(type), value_(value) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
EqualKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<EqualKernel>
EqualKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<EqualKernel>(type_, value_);
}

const Meta &EqualKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &EqualKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string EqualKernelGeneratorImpl::GetKernelName() const {
  return EqualKernel::kKernelName;
}

size_t EqualKernelGeneratorImpl::GetHashCode() const {
  std::hash<Type> type_hash;
  std::hash<int64_t> i64_hash;
  size_t hash = typeid(EqualKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= type_hash(type_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= i64_hash(value_) + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool EqualKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const EqualKernelGeneratorImpl *other_ptr =
          dynamic_cast<const EqualKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool EqualKernelGeneratorImpl::Equals(
    const EqualKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() && type_ == other.type_ &&
         value_ == other.value_;
}

} // namespace kernel
} // namespace fluidml
