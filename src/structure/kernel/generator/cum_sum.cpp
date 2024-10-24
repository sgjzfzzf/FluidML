#include "structure/kernel/generator/cum_sum.h"
#include "utils/hash.h"

namespace cpu_transformers {
namespace kernel {

class CumSumKernelGeneratorImpl : public CumSumKernelGenerator {
public:
  CumSumKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta, int64_t axis,
                            bool exclusive, bool reverse);
  CumSumKernelGeneratorImpl(const CumSumKernelGeneratorImpl &) = delete;
  CumSumKernelGeneratorImpl(CumSumKernelGeneratorImpl &&) = default;
  virtual ~CumSumKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<CumSumKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const CumSumKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const int64_t axis_;
  const bool exclusive_;
  const bool reverse_;
};

std::unique_ptr<CumSumKernelGenerator>
CumSumKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta, int64_t axis,
                            bool exclusive, bool reverse) {
  return std::make_unique<CumSumKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), axis, exclusive, reverse);
}

CumSumKernelGeneratorImpl::CumSumKernelGeneratorImpl(Meta &&input_meta,
                                                     Meta &&output_meta,
                                                     int64_t axis,
                                                     bool exclusive,
                                                     bool reverse)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      axis_(axis), exclusive_(exclusive), reverse_(reverse) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
CumSumKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<CumSumKernel>
CumSumKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                 llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<CumSumKernel>(axis_, exclusive_, reverse_);
}

const Meta &CumSumKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &CumSumKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string CumSumKernelGeneratorImpl::GetKernelName() const {
  return CumSumKernel::kKernelName;
}

size_t CumSumKernelGeneratorImpl::GetHashCode() const {
  std::hash<int64_t> i64_hash;
  std::hash<bool> bool_hash;
  size_t hash = typeid(CumSumKernelGeneratorImpl).hash_code();
  hash ^= input_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= output_meta_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= i64_hash(axis_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= bool_hash(exclusive_) + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= bool_hash(reverse_) + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool CumSumKernelGeneratorImpl::Equals(const KernelGenerator &other) const {
  if (const CumSumKernelGeneratorImpl *other_ptr =
          dynamic_cast<const CumSumKernelGeneratorImpl *>(&other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool CumSumKernelGeneratorImpl::Equals(
    const CumSumKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() && axis_ == other.axis_ &&
         exclusive_ == other.exclusive_ && reverse_ == other.reverse_;
}

} // namespace kernel
} // namespace cpu_transformers
