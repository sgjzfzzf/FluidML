#include "structure/kernel/generator/add.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class AddConstantKernelGeneratorImpl : public AddConstantKernelGenerator {
public:
  AddConstantKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                                 Type type, float64_t constant);
  AddConstantKernelGeneratorImpl(const AddConstantKernelGeneratorImpl &) =
      delete;
  AddConstantKernelGeneratorImpl(AddConstantKernelGeneratorImpl &&) = default;
  virtual ~AddConstantKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<AddConstantKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const AddConstantKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Type type_;
  const float64_t constant_;
};

class AddCommonKernelGeneratorImpl : public AddCommonKernelGenerator {
public:
  AddCommonKernelGeneratorImpl(Meta &&lhs_meta, Meta &&rhs_meta,
                               Meta &&output_meta);
  AddCommonKernelGeneratorImpl(const AddCommonKernelGeneratorImpl &) = delete;
  AddCommonKernelGeneratorImpl(AddCommonKernelGeneratorImpl &&) = default;
  virtual ~AddCommonKernelGeneratorImpl() = default;
  std::shared_ptr<DoubleInputsWithoutBufferKernel>
  YieldDoubleInputsWithoutBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<AddCommonKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetLhsMeta() const override;
  const Meta &GetRhsMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const AddCommonKernelGeneratorImpl &other) const;

private:
  const Meta lhs_meta_;
  const Meta rhs_meta_;
  const Meta output_meta_;
};

std::unique_ptr<AddConstantKernelGenerator>
AddConstantKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                                 Type type, float64_t constant) {
  return std::make_unique<AddConstantKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), type, constant);
}

std::unique_ptr<AddCommonKernelGenerator>
AddCommonKernelGenerator::Make(Meta &&lhs_meta, Meta &&rhs_meta,
                               Meta &&output_meta) {
  return std::make_unique<AddCommonKernelGeneratorImpl>(
      std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta));
}

AddConstantKernelGeneratorImpl::AddConstantKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, Type type, float64_t constant)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      type_(type), constant_(constant) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
AddConstantKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<AddConstantKernel>
AddConstantKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                      llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<AddConstantKernel>(type_, constant_);
}

const Meta &AddConstantKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &AddConstantKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string AddConstantKernelGeneratorImpl::GetKernelName() const {
  return AddConstantKernel::kKernelName;
}

size_t AddConstantKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(AddConstantKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= std::hash<Type>()(type_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^=
      std::hash<float64_t>()(constant_) + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool AddConstantKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const AddConstantKernelGeneratorImpl *other_ptr =
          dynamic_cast<const AddConstantKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool AddConstantKernelGeneratorImpl::Equals(
    const AddConstantKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ &&
         output_meta_ == other.output_meta_ && type_ == other.type_ &&
         constant_ == other.constant_;
}

AddCommonKernelGeneratorImpl::AddCommonKernelGeneratorImpl(Meta &&lhs_meta,
                                                           Meta &&rhs_meta,
                                                           Meta &&output_meta)
    : lhs_meta_(std::move(lhs_meta)), rhs_meta_(std::move(rhs_meta)),
      output_meta_(std::move(output_meta)) {}

std::shared_ptr<DoubleInputsWithoutBufferKernel>
AddCommonKernelGeneratorImpl::YieldDoubleInputsWithoutBufferKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return Yield(lhs_layout, rhs_layout, output_layout);
}

std::shared_ptr<AddCommonKernel>
AddCommonKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> lhs_layout,
                                    llvm::ArrayRef<size_t> rhs_layout,
                                    llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<AddCommonKernel>();
}

const Meta &AddCommonKernelGeneratorImpl::GetLhsMeta() const {
  return lhs_meta_;
}

const Meta &AddCommonKernelGeneratorImpl::GetRhsMeta() const {
  return rhs_meta_;
}

const Meta &AddCommonKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string AddCommonKernelGeneratorImpl::GetKernelName() const {
  return AddCommonKernel::kKernelName;
}

size_t AddCommonKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(AddCommonKernelGeneratorImpl).hash_code();
  hash ^= GetLhsMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetRhsMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool AddCommonKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const AddCommonKernelGeneratorImpl *other_ptr =
          dynamic_cast<const AddCommonKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool AddCommonKernelGeneratorImpl::Equals(
    const AddCommonKernelGeneratorImpl &other) const {
  return lhs_meta_ == other.lhs_meta_ && rhs_meta_ == other.rhs_meta_ &&
         output_meta_ == other.output_meta_;
}

} // namespace kernel
} // namespace cpu_transformers
