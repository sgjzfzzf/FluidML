#include "structure/kernel/generator/add_div_erf_add_mul_mul.h"
#include "structure/tensor/meta.h"
#include "utils/hash.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {
class AddDivErfAddMulMulKernelGeneratorImpl
    : public AddDivErfAddMulMulKernelGenerator {
public:
  AddDivErfAddMulMulKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                                        Tensor &&add0_weight, Type div_type,
                                        float64_t div_weight, Type add1_type,
                                        float64_t add1_weight, Type mul1_type,
                                        float64_t mul1_weight);
  AddDivErfAddMulMulKernelGeneratorImpl(
      const AddDivErfAddMulMulKernelGeneratorImpl &) = delete;
  AddDivErfAddMulMulKernelGeneratorImpl(
      AddDivErfAddMulMulKernelGeneratorImpl &&) = default;
  virtual ~AddDivErfAddMulMulKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<AddDivErfAddMulMulKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const AddDivErfAddMulMulKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Tensor add0_weight_;
  const Type div_type_;
  const float64_t div_weight_;
  const Type add1_type_;
  const float64_t add1_weight_;
  const Type mul1_type_;
  const float64_t mul1_weight_;
};

std::unique_ptr<AddDivErfAddMulMulKernelGenerator>
AddDivErfAddMulMulKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                                        Tensor &&add0_weight, Type div_type,
                                        float64_t div_weight, Type add1_type,
                                        float64_t add1_weight, Type mul1_type,
                                        float64_t mul1_weight) {
  return std::make_unique<AddDivErfAddMulMulKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(add0_weight),
      div_type, div_weight, add1_type, add1_weight, mul1_type, mul1_weight);
}

AddDivErfAddMulMulKernelGeneratorImpl::AddDivErfAddMulMulKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, Tensor &&add0_weight, Type div_type,
    float64_t div_weight, Type add1_type, float64_t add1_weight, Type mul1_type,
    float64_t mul1_weight)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      add0_weight_(std::move(add0_weight)), div_type_(div_type),
      div_weight_(div_weight), add1_type_(add1_type), add1_weight_(add1_weight),
      mul1_type_(mul1_type), mul1_weight_(mul1_weight) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
AddDivErfAddMulMulKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<AddDivErfAddMulMulKernel>
AddDivErfAddMulMulKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  Tensor add0_weight = add0_weight_;
  return std::make_shared<AddDivErfAddMulMulKernel>(
      std::move(add0_weight), div_type_, div_weight_, add1_type_, add1_weight_,
      mul1_type_, mul1_weight_);
}

const Meta &AddDivErfAddMulMulKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &AddDivErfAddMulMulKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string AddDivErfAddMulMulKernelGeneratorImpl::GetKernelName() const {
  return AddDivErfAddMulMulKernel::kKernelName;
}

size_t AddDivErfAddMulMulKernelGeneratorImpl::GetHashCode() const {
  std::hash<Type> type_hash;
  std::hash<float64_t> f64_hash;
  size_t hash = typeid(AddDivErfAddMulMulKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= add0_weight_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= type_hash(div_type_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= f64_hash(div_weight_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= type_hash(add1_type_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= f64_hash(add1_weight_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= type_hash(mul1_type_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= f64_hash(mul1_weight_) + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool AddDivErfAddMulMulKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const AddDivErfAddMulMulKernelGeneratorImpl *other_ptr =
          dynamic_cast<const AddDivErfAddMulMulKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool AddDivErfAddMulMulKernelGeneratorImpl::Equals(
    const AddDivErfAddMulMulKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ &&
         output_meta_ == other.output_meta_ &&
         add0_weight_ == other.add0_weight_ && div_type_ == other.div_type_ &&
         div_weight_ == other.div_weight_ && add1_type_ == other.add1_type_ &&
         add1_weight_ == other.add1_weight_ && mul1_type_ == other.mul1_type_ &&
         mul1_weight_ == other.mul1_weight_;
}

} // namespace kernel
} // namespace cpu_transformers