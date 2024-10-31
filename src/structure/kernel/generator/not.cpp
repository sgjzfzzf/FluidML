#include "structure/kernel/generator/not.h"
#include "structure/kernel/kernel/not.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class NotKernelGeneratorImpl : public NotKernelGenerator {
public:
  NotKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta);
  NotKernelGeneratorImpl(const NotKernelGeneratorImpl &) = delete;
  NotKernelGeneratorImpl(NotKernelGeneratorImpl &&) = default;
  virtual ~NotKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<NotKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const NotKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
};

std::unique_ptr<NotKernelGenerator>
NotKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta) {
  return std::make_unique<NotKernelGeneratorImpl>(std::move(input_meta),
                                                  std::move(output_meta));
}

NotKernelGeneratorImpl::NotKernelGeneratorImpl(Meta &&input_meta,
                                               Meta &&output_meta)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)) {
}

std::shared_ptr<SingleInputWithoutBufferKernel>
NotKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<NotKernel>
NotKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                              llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<NotKernel>();
}

const Meta &NotKernelGeneratorImpl::GetInputMeta() const { return input_meta_; }

const Meta &NotKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string NotKernelGeneratorImpl::GetKernelName() const {
  return NotKernel::kKernelName;
}

size_t NotKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(NotKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool NotKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const NotKernelGeneratorImpl *other_ptr =
          dynamic_cast<const NotKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool NotKernelGeneratorImpl::Equals(const NotKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta();
}

} // namespace kernel
} // namespace fluidml
