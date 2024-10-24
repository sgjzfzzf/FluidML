#include "structure/kernel/generator/tanh.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class TanhKernelGeneratorImpl : public TanhKernelGenerator {
public:
  TanhKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta);
  TanhKernelGeneratorImpl(const TanhKernelGeneratorImpl &) = delete;
  TanhKernelGeneratorImpl(TanhKernelGeneratorImpl &&) = default;
  virtual ~TanhKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<TanhKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const TanhKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
};

std::unique_ptr<TanhKernelGenerator>
TanhKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta) {
  return std::make_unique<TanhKernelGeneratorImpl>(std::move(input_meta),
                                                   std::move(output_meta));
}

TanhKernelGeneratorImpl::TanhKernelGeneratorImpl(Meta &&input_meta,
                                                 Meta &&output_meta)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)) {
}

std::shared_ptr<SingleInputWithoutBufferKernel>
TanhKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<TanhKernel>
TanhKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                               llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<TanhKernel>();
}

const Meta &TanhKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &TanhKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string TanhKernelGeneratorImpl::GetKernelName() const {
  return TanhKernel::kKernelName;
}

size_t TanhKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(TanhKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool TanhKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const TanhKernelGeneratorImpl *other_ptr =
          dynamic_cast<const TanhKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool TanhKernelGeneratorImpl::Equals(
    const TanhKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ && output_meta_ == other.output_meta_;
}

} // namespace kernel
} // namespace cpu_transformers
