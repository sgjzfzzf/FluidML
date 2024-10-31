#include "structure/kernel/generator/relu.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class ReluKernelGeneratorImpl : public ReluKernelGenerator {
public:
  ReluKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta);
  ReluKernelGeneratorImpl(const ReluKernelGeneratorImpl &) = delete;
  ReluKernelGeneratorImpl(ReluKernelGeneratorImpl &&) = default;
  virtual ~ReluKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<ReluKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const ReluKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
};

std::unique_ptr<ReluKernelGenerator>
ReluKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta) {
  return std::make_unique<ReluKernelGeneratorImpl>(std::move(input_meta),
                                                   std::move(output_meta));
}

ReluKernelGeneratorImpl::ReluKernelGeneratorImpl(Meta &&input_meta,
                                                 Meta &&output_meta)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)) {
}

std::shared_ptr<SingleInputWithoutBufferKernel>
ReluKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<ReluKernel>
ReluKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                               llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<ReluKernel>();
}

const Meta &ReluKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &ReluKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string ReluKernelGeneratorImpl::GetKernelName() const {
  return ReluKernel::kKernelName;
}

size_t ReluKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(ReluKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool ReluKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const ReluKernelGeneratorImpl *other_ptr =
          dynamic_cast<const ReluKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool ReluKernelGeneratorImpl::Equals(
    const ReluKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta();
}

} // namespace kernel
} // namespace cpu_transformers
