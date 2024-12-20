#include "structure/kernel/generator/pow.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class PowKernelGeneratorImpl : public PowKernelGenerator {
public:
  PowKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta, Type type,
                         float64_t exp);
  PowKernelGeneratorImpl(const PowKernelGeneratorImpl &) = delete;
  PowKernelGeneratorImpl(PowKernelGeneratorImpl &&) = default;
  virtual ~PowKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<PowKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const PowKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Type type_;
  const float64_t exp_;
};

std::unique_ptr<PowKernelGenerator> PowKernelGenerator::Make(Meta &&input_meta,
                                                             Meta &&output_meta,
                                                             Type type,
                                                             float64_t exp) {
  return std::make_unique<PowKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), type, exp);
}

PowKernelGeneratorImpl::PowKernelGeneratorImpl(Meta &&input_meta,
                                               Meta &&output_meta, Type type,
                                               float64_t exp)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      type_(type), exp_(exp) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
PowKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<PowKernel>
PowKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                              llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<PowKernel>(type_, exp_);
}

const Meta &PowKernelGeneratorImpl::GetInputMeta() const { return input_meta_; }

const Meta &PowKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string PowKernelGeneratorImpl::GetKernelName() const {
  return PowKernel::kKernelName;
}

size_t PowKernelGeneratorImpl::GetHashCode() const {
  std::hash<Type> type_hash;
  std::hash<float64_t> f64_hash;
  size_t hash = typeid(PowKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= type_hash(type_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= f64_hash(exp_) + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool PowKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const PowKernelGeneratorImpl *other_ptr =
          dynamic_cast<const PowKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool PowKernelGeneratorImpl::Equals(const PowKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ &&
         output_meta_ == other.output_meta_ && type_ == other.type_ &&
         exp_ == other.exp_;
}

} // namespace kernel
} // namespace fluidml
