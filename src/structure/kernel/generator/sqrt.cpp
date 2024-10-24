#include "structure/kernel/generator/sqrt.h"

namespace cpu_transformers {
namespace kernel {

class SqrtKernelGeneratorImpl : public SqrtKernelGenerator {
public:
  SqrtKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta);
  SqrtKernelGeneratorImpl(const SqrtKernelGeneratorImpl &) = delete;
  SqrtKernelGeneratorImpl(SqrtKernelGeneratorImpl &&) = default;
  virtual ~SqrtKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<SqrtKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const SqrtKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
};

std::unique_ptr<SqrtKernelGenerator>
SqrtKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta) {
  return std::make_unique<SqrtKernelGeneratorImpl>(std::move(input_meta),
                                                   std::move(output_meta));
}

SqrtKernelGeneratorImpl::SqrtKernelGeneratorImpl(Meta &&input_meta,
                                                 Meta &&output_meta)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)) {
}

std::shared_ptr<SingleInputWithoutBufferKernel>
SqrtKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<SqrtKernel>
SqrtKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                               llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<SqrtKernel>();
}

const Meta &SqrtKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &SqrtKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string SqrtKernelGeneratorImpl::GetKernelName() const {
  return SqrtKernel::kKernelName;
}

size_t SqrtKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(SqrtKernelGeneratorImpl).hash_code();
  hash ^= input_meta_.GetHashCode();
  hash ^= output_meta_.GetHashCode();
  return hash;
}

bool SqrtKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const SqrtKernelGeneratorImpl *other_ptr =
          dynamic_cast<const SqrtKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool SqrtKernelGeneratorImpl::Equals(
    const SqrtKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetOutputMeta() &&
         GetInputMeta() == other.GetOutputMeta();
}

} // namespace kernel
} // namespace cpu_transformers
