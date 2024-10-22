#include "structure/kernel/generator/unsqueeze.h"

namespace cpu_transformers {
namespace kernel {

class UnSqueezeKernelGeneratorImpl : public UnSqueezeKernelGenerator {
public:
  UnSqueezeKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                               std::vector<int64_t> &&axes);
  UnSqueezeKernelGeneratorImpl(const UnSqueezeKernelGeneratorImpl &) = delete;
  UnSqueezeKernelGeneratorImpl(UnSqueezeKernelGeneratorImpl &&) = default;
  virtual ~UnSqueezeKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<UnSqueezeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  virtual size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const UnSqueezeKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const std::vector<int64_t> axes_;
};

std::unique_ptr<UnSqueezeKernelGenerator>
UnSqueezeKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                               std::vector<int64_t> &&axes) {
  return std::make_unique<UnSqueezeKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(axes));
}

UnSqueezeKernelGeneratorImpl::UnSqueezeKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&axes)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      axes_(std::move(axes)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
UnSqueezeKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<UnSqueezeKernel>
UnSqueezeKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                    llvm::ArrayRef<size_t> output_layout) {
  std::vector<int64_t> axes = axes_;
  return std::make_shared<UnSqueezeKernel>(std::move(axes));
}

const Meta &UnSqueezeKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &UnSqueezeKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string UnSqueezeKernelGeneratorImpl::GetKernelName() const {
  return UnSqueezeKernel::kKernelName;
}

size_t UnSqueezeKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(UnSqueezeKernelGeneratorImpl).hash_code();
  std::hash<int64_t> i64_hash;
  hash ^= input_meta_.GetHashCode() + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= output_meta_.GetHashCode() + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  for (int64_t axis : axes_) {
    hash ^= i64_hash(axis) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }
  return hash;
}

bool UnSqueezeKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const UnSqueezeKernelGeneratorImpl *other_ptr =
          dynamic_cast<const UnSqueezeKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool UnSqueezeKernelGeneratorImpl::Equals(
    const UnSqueezeKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ &&
         output_meta_ == other.output_meta_ && axes_ == other.axes_;
}

} // namespace kernel
} // namespace cpu_transformers