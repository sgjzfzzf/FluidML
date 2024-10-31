#include "structure/kernel/generator/unsqueeze.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class UnsqueezeKernelGeneratorImpl : public UnsqueezeKernelGenerator {
public:
  UnsqueezeKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                               std::vector<int64_t> &&axes);
  UnsqueezeKernelGeneratorImpl(const UnsqueezeKernelGeneratorImpl &) = delete;
  UnsqueezeKernelGeneratorImpl(UnsqueezeKernelGeneratorImpl &&) = default;
  virtual ~UnsqueezeKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<UnsqueezeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  virtual size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const UnsqueezeKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const std::vector<int64_t> axes_;
};

std::unique_ptr<UnsqueezeKernelGenerator>
UnsqueezeKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                               std::vector<int64_t> &&axes) {
  return std::make_unique<UnsqueezeKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(axes));
}

UnsqueezeKernelGeneratorImpl::UnsqueezeKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&axes)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      axes_(std::move(axes)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
UnsqueezeKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<UnsqueezeKernel>
UnsqueezeKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                    llvm::ArrayRef<size_t> output_layout) {
  std::vector<int64_t> axes = axes_;
  return std::make_shared<UnsqueezeKernel>(std::move(axes));
}

const Meta &UnsqueezeKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &UnsqueezeKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string UnsqueezeKernelGeneratorImpl::GetKernelName() const {
  return UnsqueezeKernel::kKernelName;
}

size_t UnsqueezeKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(UnsqueezeKernelGeneratorImpl).hash_code();
  std::hash<int64_t> i64_hash;
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  for (int64_t axis : axes_) {
    hash ^= i64_hash(axis) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  return hash;
}

bool UnsqueezeKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const UnsqueezeKernelGeneratorImpl *other_ptr =
          dynamic_cast<const UnsqueezeKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool UnsqueezeKernelGeneratorImpl::Equals(
    const UnsqueezeKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ &&
         output_meta_ == other.output_meta_ && axes_ == other.axes_;
}

} // namespace kernel
} // namespace fluidml