#include "structure/kernel/generator/pad.h"
#include "structure/kernel/kernel/kernel.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class PadKernelGeneratorImpl : public PadKernelGenerator {
public:
  PadKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                         std::vector<std::tuple<int64_t, int64_t>> &&pads);
  PadKernelGeneratorImpl(const PadKernelGeneratorImpl &) = delete;
  PadKernelGeneratorImpl(PadKernelGeneratorImpl &&) = default;
  virtual ~PadKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  virtual std::shared_ptr<PadKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const PadKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const std::vector<std::tuple<int64_t, int64_t>> pads_;
};

std::unique_ptr<PadKernelGenerator>
PadKernelGenerator::Make(Meta &&lhs_meta, Meta &&output_meta,
                         std::vector<std::tuple<int64_t, int64_t>> &&pads) {
  return std::make_unique<PadKernelGeneratorImpl>(
      std::move(lhs_meta), std::move(output_meta), std::move(pads));
}

PadKernelGeneratorImpl::PadKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta,
    std::vector<std::tuple<int64_t, int64_t>> &&pads)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      pads_(std::move(pads)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
PadKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<PadKernel>
PadKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                              llvm::ArrayRef<size_t> output_layout) {
  std::vector<std::tuple<int64_t, int64_t>> pads = pads_;
  return std::make_shared<PadKernel>(std::move(pads));
}

const Meta &PadKernelGeneratorImpl::GetInputMeta() const { return input_meta_; }

const Meta &PadKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string PadKernelGeneratorImpl::GetKernelName() const {
  return PadKernel::kKernelName;
}

size_t PadKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(PadKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool PadKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const PadKernelGeneratorImpl *other_ptr =
          dynamic_cast<const PadKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool PadKernelGeneratorImpl::Equals(const PadKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() && pads_ == other.pads_;
}

} // namespace kernel
} // namespace fluidml
