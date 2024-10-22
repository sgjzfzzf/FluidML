#include "structure/kernel/generator/concat.h"
#include "structure/kernel/kernel/concat.h"
#include "utils/hash.h"
#include <memory>

namespace cpu_transformers {
namespace kernel {

class Concat2KernelGeneratorImpl : public Concat2KernelGenerator {
public:
  Concat2KernelGeneratorImpl(Meta &&lhs_meta, Meta &&rhs_meta,
                             Meta &&output_meta, size_t axis);
  Concat2KernelGeneratorImpl(const Concat2KernelGeneratorImpl &) = delete;
  Concat2KernelGeneratorImpl(Concat2KernelGeneratorImpl &&) = default;
  virtual ~Concat2KernelGeneratorImpl() = default;
  std::shared_ptr<DoubleInputsWithoutBufferKernel>
  YieldDoubleInputsWithoutBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<Concat2Kernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetLhsMeta() const override;
  const Meta &GetRhsMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const Concat2KernelGeneratorImpl &other) const;

private:
  const Meta lhs_meta_;
  const Meta rhs_meta_;
  const Meta output_meta_;
  const size_t axis_;
};

std::unique_ptr<Concat2KernelGenerator>
Concat2KernelGenerator::Make(Meta &&lhs_meta, Meta &&rhs_meta,
                             Meta &&output_meta, size_t axis) {
  return std::make_unique<Concat2KernelGeneratorImpl>(
      std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta), axis);
}

Concat2KernelGeneratorImpl::Concat2KernelGeneratorImpl(Meta &&lhs_meta,
                                                       Meta &&rhs_meta,
                                                       Meta &&output_meta,
                                                       size_t axis)
    : lhs_meta_(std::move(lhs_meta)), rhs_meta_(std::move(rhs_meta)),
      output_meta_(std::move(output_meta)), axis_(axis) {}

std::shared_ptr<DoubleInputsWithoutBufferKernel>
Concat2KernelGeneratorImpl::YieldDoubleInputsWithoutBufferKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return Yield(lhs_layout, rhs_layout, output_layout);
}

std::shared_ptr<Concat2Kernel>
Concat2KernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> lhs_layout,
                                  llvm::ArrayRef<size_t> rhs_layout,
                                  llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<Concat2Kernel>(axis_);
}

const Meta &Concat2KernelGeneratorImpl::GetLhsMeta() const { return lhs_meta_; }

const Meta &Concat2KernelGeneratorImpl::GetRhsMeta() const { return rhs_meta_; }

const Meta &Concat2KernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string Concat2KernelGeneratorImpl::GetKernelName() const {
  return Concat2Kernel::kKernelName;
}

size_t Concat2KernelGeneratorImpl::GetHashCode() const {
  std::hash<size_t> u64_hash;
  size_t hash = typeid(Concat2KernelGeneratorImpl).hash_code();
  hash ^= lhs_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= rhs_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= output_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= u64_hash(axis_) + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool Concat2KernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (typeid(*this) != typeid(other)) {
    return false;
  }
  return Equals(static_cast<const Concat2KernelGeneratorImpl &>(other));
}

bool Concat2KernelGeneratorImpl::Equals(
    const Concat2KernelGeneratorImpl &other) const {
  return GetLhsMeta() == other.GetLhsMeta() &&
         GetRhsMeta() == other.GetRhsMeta() &&
         GetOutputMeta() == other.GetOutputMeta() && axis_ == other.axis_;
}

} // namespace kernel
} // namespace cpu_transformers
