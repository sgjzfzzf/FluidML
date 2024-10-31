#include "structure/kernel/generator/flatten.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class FlattenKernelGeneratorImpl : public FlattenKernelGenerator {
public:
  FlattenKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                             int64_t axis);
  FlattenKernelGeneratorImpl(const FlattenKernelGeneratorImpl &) = delete;
  FlattenKernelGeneratorImpl(FlattenKernelGeneratorImpl &&) = default;
  virtual ~FlattenKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<FlattenKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const FlattenKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const int64_t axis_;
};

std::unique_ptr<FlattenKernelGenerator>
FlattenKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                             int64_t axis) {
  return std::make_unique<FlattenKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), axis);
}

FlattenKernelGeneratorImpl::FlattenKernelGeneratorImpl(Meta &&input_meta,
                                                       Meta &&output_meta,
                                                       int64_t axis)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      axis_(axis) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
FlattenKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<FlattenKernel>
FlattenKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                  llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<FlattenKernel>(axis_);
}

const Meta &FlattenKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &FlattenKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string FlattenKernelGeneratorImpl::GetKernelName() const {
  return FlattenKernel::kKernelName;
}

size_t FlattenKernelGeneratorImpl::GetHashCode() const {
  std::hash<int64_t> i64_hasher;
  size_t hash = typeid(FlattenKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= i64_hasher(axis_) + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool FlattenKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const FlattenKernelGeneratorImpl *p =
          dynamic_cast<const FlattenKernelGeneratorImpl *>(&other)) {
    return Equals(*p);
  } else {
    return false;
  }
}

bool FlattenKernelGeneratorImpl::Equals(
    const FlattenKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() && axis_ == other.axis_;
}

} // namespace kernel
} // namespace fluidml
