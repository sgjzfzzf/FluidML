#include "structure/kernel/generator/cast.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class CastKernelGeneratorImpl : public CastKernelGenerator {
public:
  CastKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta);
  CastKernelGeneratorImpl(const CastKernelGeneratorImpl &) = delete;
  CastKernelGeneratorImpl(CastKernelGeneratorImpl &&) = default;
  virtual ~CastKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<CastKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const CastKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
};

std::unique_ptr<CastKernelGenerator>
CastKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta) {
  return std::make_unique<CastKernelGeneratorImpl>(std::move(input_meta),
                                                   std::move(output_meta));
}

CastKernelGeneratorImpl::CastKernelGeneratorImpl(Meta &&input_meta,
                                                 Meta &&output_meta)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)) {
}

std::shared_ptr<SingleInputWithoutBufferKernel>
CastKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<CastKernel>
CastKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                               llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<CastKernel>();
}

const Meta &CastKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &CastKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string CastKernelGeneratorImpl::GetKernelName() const {
  return CastKernel::kKernelName;
}

size_t CastKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(CastKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool CastKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const CastKernelGeneratorImpl *other_ptr =
          dynamic_cast<const CastKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool CastKernelGeneratorImpl::Equals(
    const CastKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta();
}

} // namespace kernel
} // namespace fluidml
