#include "structure/kernel/generator/dropout.h"
#include "utils/float.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class DropoutKernelGeneratorImpl : public DropoutKernelGenerator {
public:
  DropoutKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                             float64_t ratio);
  DropoutKernelGeneratorImpl(const DropoutKernelGeneratorImpl &) = delete;
  DropoutKernelGeneratorImpl(DropoutKernelGeneratorImpl &&) = default;
  virtual ~DropoutKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<DropoutKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const DropoutKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const float64_t ratio_;
};

std::unique_ptr<DropoutKernelGenerator>
DropoutKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                             float64_t ratio) {
  return std::make_unique<DropoutKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), ratio);
}

DropoutKernelGeneratorImpl::DropoutKernelGeneratorImpl(Meta &&input_meta,
                                                       Meta &&output_meta,
                                                       float64_t ratio)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      ratio_(ratio) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
DropoutKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<DropoutKernel>
DropoutKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                  llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<DropoutKernel>(ratio_);
}

const Meta &DropoutKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &DropoutKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string DropoutKernelGeneratorImpl::GetKernelName() const {
  return DropoutKernel::kKernelName;
}

size_t DropoutKernelGeneratorImpl::GetHashCode() const {
  std::hash<float64_t> f64_hasher;
  size_t hash = typeid(DropoutKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= f64_hasher(ratio_) + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool DropoutKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const DropoutKernelGeneratorImpl *p =
          dynamic_cast<const DropoutKernelGeneratorImpl *>(&other)) {
    return Equals(*p);
  } else {
    return false;
  }
}

bool DropoutKernelGeneratorImpl::Equals(
    const DropoutKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() && ratio_ == other.ratio_;
}

} // namespace kernel
} // namespace fluidml
