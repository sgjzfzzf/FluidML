#include "structure/kernel/generator/mul.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class MulConstantKernelGeneratorImpl : public MulConstantKernelGenerator {
public:
  MulConstantKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                                 Type type, float64_t constant);
  MulConstantKernelGeneratorImpl(const MulConstantKernelGeneratorImpl &) =
      delete;
  MulConstantKernelGeneratorImpl(MulConstantKernelGeneratorImpl &&) = default;
  virtual ~MulConstantKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<MulConstantKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const MulConstantKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Type type_;
  const float64_t constant_;
};

class MulCommonKernelGeneratorImpl : public MulCommonKernelGenerator {
public:
  MulCommonKernelGeneratorImpl(Meta &&lhs_meta, Meta &&rhs_meta,
                               Meta &&output_meta);
  MulCommonKernelGeneratorImpl(const MulCommonKernelGeneratorImpl &) = delete;
  MulCommonKernelGeneratorImpl(MulCommonKernelGeneratorImpl &&) = default;
  virtual ~MulCommonKernelGeneratorImpl() = default;
  std::shared_ptr<DoubleInputsWithoutBufferKernel>
  YieldDoubleInputsWithoutBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<MulCommonKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetLhsMeta() const override;
  const Meta &GetRhsMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const MulCommonKernelGeneratorImpl &other) const;

private:
  const Meta lhs_meta_;
  const Meta rhs_meta_;
  const Meta output_meta_;
};

std::unique_ptr<MulConstantKernelGenerator>
MulConstantKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                                 Type type, float64_t constant) {
  return std::make_unique<MulConstantKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), type, constant);
}

std::unique_ptr<MulCommonKernelGenerator>
MulCommonKernelGenerator::Make(Meta &&lhs_meta, Meta &&rhs_meta,
                               Meta &&output_meta) {
  return std::make_unique<MulCommonKernelGeneratorImpl>(
      std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta));
}

MulConstantKernelGeneratorImpl::MulConstantKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, Type type, float64_t constant)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      type_(type), constant_(constant) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
MulConstantKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<MulConstantKernel>
MulConstantKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                      llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<MulConstantKernel>(type_, constant_);
}

const Meta &MulConstantKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &MulConstantKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string MulConstantKernelGeneratorImpl::GetKernelName() const {
  return MulConstantKernel::kKernelName;
}

size_t MulConstantKernelGeneratorImpl::GetHashCode() const {
  std::hash<Type> type_hash;
  std::hash<float64_t> f64_hash;
  size_t hash = typeid(MulConstantKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= type_hash(type_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= f64_hash(constant_) + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool MulConstantKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const MulConstantKernelGeneratorImpl *other_ptr =
          dynamic_cast<const MulConstantKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool MulConstantKernelGeneratorImpl::Equals(
    const MulConstantKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ &&
         output_meta_ == other.output_meta_ && type_ == other.type_ &&
         constant_ == other.constant_;
}

MulCommonKernelGeneratorImpl::MulCommonKernelGeneratorImpl(Meta &&lhs_meta,
                                                           Meta &&rhs_meta,
                                                           Meta &&output_meta)
    : lhs_meta_(std::move(lhs_meta)), rhs_meta_(std::move(rhs_meta)),
      output_meta_(std::move(output_meta)) {}

std::shared_ptr<DoubleInputsWithoutBufferKernel>
MulCommonKernelGeneratorImpl::YieldDoubleInputsWithoutBufferKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return std::make_unique<MulCommonKernel>();
}

std::shared_ptr<MulCommonKernel>
MulCommonKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> lhs_layout,
                                    llvm::ArrayRef<size_t> rhs_layout,
                                    llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<MulCommonKernel>();
}

const Meta &MulCommonKernelGeneratorImpl::GetLhsMeta() const {
  return lhs_meta_;
}

const Meta &MulCommonKernelGeneratorImpl::GetRhsMeta() const {
  return rhs_meta_;
}

const Meta &MulCommonKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string MulCommonKernelGeneratorImpl::GetKernelName() const {
  return MulCommonKernel::kKernelName;
}

size_t MulCommonKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(MulCommonKernelGeneratorImpl).hash_code();
  hash ^= GetLhsMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetRhsMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool MulCommonKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const MulCommonKernelGeneratorImpl *other_ptr =
          dynamic_cast<const MulCommonKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool MulCommonKernelGeneratorImpl::Equals(
    const MulCommonKernelGeneratorImpl &other) const {
  return lhs_meta_ == other.lhs_meta_ && rhs_meta_ == other.rhs_meta_ &&
         output_meta_ == other.output_meta_;
}

} // namespace kernel
} // namespace cpu_transformers
