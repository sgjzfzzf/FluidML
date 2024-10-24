#include "structure/kernel/generator/reduce_mean.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class ReduceMeanKernelGeneratorImpl : public ReduceMeanKernelGenerator {
public:
  ReduceMeanKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                                llvm::SmallVector<int64_t> &&axes,
                                bool keep_dims);
  ReduceMeanKernelGeneratorImpl(const ReduceMeanKernelGeneratorImpl &) = delete;
  ReduceMeanKernelGeneratorImpl(ReduceMeanKernelGeneratorImpl &&) = default;
  virtual ~ReduceMeanKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<ReduceMeanKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const ReduceMeanKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const llvm::SmallVector<int64_t> axes_;
  const bool keep_dims_;
};

std::unique_ptr<ReduceMeanKernelGenerator>
ReduceMeanKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                                llvm::SmallVector<int64_t> &&axes,
                                bool keep_dims) {
  return std::make_unique<ReduceMeanKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(axes),
      keep_dims);
}

ReduceMeanKernelGeneratorImpl::ReduceMeanKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, llvm::SmallVector<int64_t> &&axes,
    bool keep_dims)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      axes_(std::move(axes)), keep_dims_(keep_dims) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
ReduceMeanKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<ReduceMeanKernel>
ReduceMeanKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                     llvm::ArrayRef<size_t> output_layout) {
  llvm::SmallVector<int64_t> axes = axes_;
  return std::make_shared<ReduceMeanKernel>(std::move(axes), keep_dims_);
}

const Meta &ReduceMeanKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &ReduceMeanKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string ReduceMeanKernelGeneratorImpl::GetKernelName() const {
  return ReduceMeanKernel::kKernelName;
}

size_t ReduceMeanKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(ReduceMeanKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool ReduceMeanKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const ReduceMeanKernelGeneratorImpl *other_ptr =
          dynamic_cast<const ReduceMeanKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool ReduceMeanKernelGeneratorImpl::Equals(
    const ReduceMeanKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() && axes_ == other.axes_ &&
         keep_dims_ == other.keep_dims_;
}

} // namespace kernel
} // namespace cpu_transformers
