#include "structure/kernel/generator/clip.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class ClipKernelGeneratorImpl : public ClipKernelGenerator {
public:
  ClipKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta, float32_t min,
                          float32_t max);
  ClipKernelGeneratorImpl(const ClipKernelGeneratorImpl &) = delete;
  ClipKernelGeneratorImpl(ClipKernelGeneratorImpl &&) = default;
  virtual ~ClipKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<ClipKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const ClipKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const float32_t min_;
  const float32_t max_;
};

std::unique_ptr<ClipKernelGenerator>
ClipKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta, float32_t min,
                          float32_t max) {
  return std::make_unique<ClipKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), min, max);
}

ClipKernelGeneratorImpl::ClipKernelGeneratorImpl(Meta &&input_meta,
                                                 Meta &&output_meta,
                                                 float32_t min, float32_t max)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      min_(min), max_(max) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
ClipKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<ClipKernel>
ClipKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                               llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<ClipKernel>(min_, max_);
}

const Meta &ClipKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &ClipKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string ClipKernelGeneratorImpl::GetKernelName() const {
  return ClipKernel::kKernelName;
}

size_t ClipKernelGeneratorImpl::GetHashCode() const {
  std::hash<float32_t> hasher;
  size_t hash = typeid(ClipKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= hasher(min_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= hasher(max_) + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool ClipKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  return Equals(dynamic_cast<const ClipKernelGeneratorImpl &>(other));
}

bool ClipKernelGeneratorImpl::Equals(
    const ClipKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() && min_ == other.min_ &&
         max_ == other.max_;
}

} // namespace kernel
} // namespace fluidml
