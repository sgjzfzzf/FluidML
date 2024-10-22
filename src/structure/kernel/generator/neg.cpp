#include "structure/kernel/generator/neg.h"
#include "structure/kernel/kernel/neg.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class NegKernelGeneratorImpl : public NegKernelGenerator {
public:
  NegKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta);
  NegKernelGeneratorImpl(const NegKernelGeneratorImpl &generator) = delete;
  NegKernelGeneratorImpl(NegKernelGeneratorImpl &&generator) = default;
  virtual ~NegKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<NegKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const NegKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
};

std::unique_ptr<NegKernelGenerator>
NegKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta) {
  return std::make_unique<NegKernelGeneratorImpl>(std::move(input_meta),
                                                  std::move(output_meta));
}

NegKernelGeneratorImpl::NegKernelGeneratorImpl(Meta &&input_meta,
                                               Meta &&output_meta)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)) {
}

std::shared_ptr<SingleInputWithoutBufferKernel>
NegKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<NegKernel>
NegKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                              llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<NegKernel>();
}

const Meta &NegKernelGeneratorImpl::GetInputMeta() const { return input_meta_; }

const Meta &NegKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string NegKernelGeneratorImpl::GetKernelName() const {
  return NegKernel::kKernelName;
}

size_t NegKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(NegKernelGeneratorImpl).hash_code();
  hash ^= input_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= output_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool NegKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (typeid(*this) != typeid(other)) {
    return false;
  }
  return Equals(static_cast<const NegKernelGeneratorImpl &>(other));
}

bool NegKernelGeneratorImpl::Equals(const NegKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta();
}

} // namespace kernel
} // namespace cpu_transformers
