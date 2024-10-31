#include "structure/kernel/generator/slice.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class SliceKernelGeneratorImpl : public SliceKernelGenerator {
public:
  SliceKernelGeneratorImpl(
      Meta &&input_meta, Meta &&output_meta,
      llvm::SmallVector<llvm::SmallVector<int64_t, 4>> &&informations);
  SliceKernelGeneratorImpl(const SliceKernelGeneratorImpl &) = delete;
  SliceKernelGeneratorImpl(SliceKernelGeneratorImpl &&) = default;
  virtual ~SliceKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<SliceKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const SliceKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const llvm::SmallVector<llvm::SmallVector<int64_t, 4>> informations_;
};

std::unique_ptr<SliceKernelGenerator> SliceKernelGenerator::Make(
    Meta &&input_meta, Meta &&output_meta,
    llvm::SmallVector<llvm::SmallVector<int64_t, 4>> &&informations) {
  return std::make_unique<SliceKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(informations));
}

SliceKernelGeneratorImpl::SliceKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta,
    llvm::SmallVector<llvm::SmallVector<int64_t, 4>> &&informations)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      informations_(std::move(informations)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
SliceKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<SliceKernel>
SliceKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                llvm::ArrayRef<size_t> output_layout) {
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>> informations = informations_;
  return std::make_shared<SliceKernel>(std::move(informations));
}

const Meta &SliceKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &SliceKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string SliceKernelGeneratorImpl::GetKernelName() const {
  return SliceKernel::kKernelName;
}

size_t SliceKernelGeneratorImpl::GetHashCode() const {
  std::hash<int64_t> hasher;
  size_t hash = typeid(SliceKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  for (llvm::ArrayRef information : informations_) {
    for (int64_t value : information) {
      hash ^= hasher(value) + kHashSeed + (hash << 6) + (hash >> 2);
    }
  }
  return hash;
}

bool SliceKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (typeid(*this) != typeid(other)) {
    return false;
  }
  return Equals(static_cast<const SliceKernelGeneratorImpl &>(other));
}

bool SliceKernelGeneratorImpl::Equals(
    const SliceKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() &&
         informations_ == other.informations_;
}

} // namespace kernel
} // namespace fluidml
