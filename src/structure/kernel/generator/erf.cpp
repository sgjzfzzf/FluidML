#include "structure/kernel/generator/erf.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class ErfKernelGeneratorImpl : public ErfKernelGenerator {
public:
  ErfKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta);
  ErfKernelGeneratorImpl(const ErfKernelGeneratorImpl &generator) = delete;
  ErfKernelGeneratorImpl(ErfKernelGeneratorImpl &&generator) = default;
  virtual ~ErfKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<ErfKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const ErfKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
};

std::unique_ptr<ErfKernelGenerator>
ErfKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta) {
  return std::make_unique<ErfKernelGeneratorImpl>(std::move(input_meta),
                                                  std::move(output_meta));
}

ErfKernelGeneratorImpl::ErfKernelGeneratorImpl(Meta &&input_meta,
                                               Meta &&output_meta)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)) {
}

std::shared_ptr<SingleInputWithoutBufferKernel>
ErfKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<ErfKernel>
ErfKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                              llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<ErfKernel>();
}

const Meta &ErfKernelGeneratorImpl::GetInputMeta() const { return input_meta_; }

const Meta &ErfKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string ErfKernelGeneratorImpl::GetKernelName() const {
  return ErfKernel::kKernelName;
}

size_t ErfKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(ErfKernelGeneratorImpl).hash_code();
  hash ^= input_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= output_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool ErfKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const ErfKernelGeneratorImpl *other_ptr =
          dynamic_cast<const ErfKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool ErfKernelGeneratorImpl::Equals(const ErfKernelGeneratorImpl &other) const {
  return input_meta_ == other.input_meta_ && output_meta_ == other.output_meta_;
}

} // namespace kernel
} // namespace cpu_transformers
