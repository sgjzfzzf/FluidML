#include "structure/kernel/generator/squeeze.h"
#include "structure/kernel/generator/generator.h"
#include "utils/hash.h"
#include <cstdint>

namespace fluidml {
namespace kernel {

class SqueezeKernelGeneratorImpl : public SqueezeKernelGenerator {
public:
  SqueezeKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                             std::vector<int64_t> &&axes);
  SqueezeKernelGeneratorImpl(const SqueezeKernelGeneratorImpl &) = delete;
  SqueezeKernelGeneratorImpl(SqueezeKernelGeneratorImpl &&) = default;
  virtual ~SqueezeKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<SqueezeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const SqueezeKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const std::vector<int64_t> axes_;
};

std::unique_ptr<SqueezeKernelGenerator>
SqueezeKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                             std::vector<int64_t> &&axes) {
  return std::make_unique<SqueezeKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(axes));
}

SqueezeKernelGeneratorImpl::SqueezeKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&axes)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      axes_(std::move(axes)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
SqueezeKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<SqueezeKernel>
SqueezeKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                  llvm::ArrayRef<size_t> output_layout) {
  std::vector<int64_t> axes = axes_;
  return std::make_shared<SqueezeKernel>(std::move(axes));
}

const Meta &SqueezeKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &SqueezeKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string SqueezeKernelGeneratorImpl::GetKernelName() const {
  return SqueezeKernel::kKernelName;
}

size_t SqueezeKernelGeneratorImpl::GetHashCode() const {
  std::hash<int64_t> i64_hash;
  size_t hash = typeid(SqueezeKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  for (auto &axis : axes_) {
    hash ^= i64_hash(axis) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  return hash;
}

bool SqueezeKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const SqueezeKernelGeneratorImpl *other_ptr =
          dynamic_cast<const SqueezeKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  }
  return false;
}

bool SqueezeKernelGeneratorImpl::Equals(
    const SqueezeKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() && axes_ == other.axes_;
}

} // namespace kernel
} // namespace fluidml
