#include "structure/kernel/generator/gather.h"
#include "utils/hash.h"

namespace fluidml {
namespace kernel {

class GatherConstantIndexScalarKernelGeneratorImpl
    : public GatherConstantIndexScalarKernelGenerator {
public:
  GatherConstantIndexScalarKernelGeneratorImpl(Meta &&input_meta,
                                               Meta &&output_meta, int64_t axis,
                                               int64_t index);
  GatherConstantIndexScalarKernelGeneratorImpl(
      const GatherConstantIndexScalarKernelGeneratorImpl &) = delete;
  GatherConstantIndexScalarKernelGeneratorImpl(
      GatherConstantIndexScalarKernelGeneratorImpl &&) = default;
  virtual ~GatherConstantIndexScalarKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<GatherConstantIndexScalarKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const GatherConstantIndexScalarKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const int64_t axis_;
  const int64_t index_;
};

class GatherConstantIndicesTensorKernelGeneratorImpl
    : public GatherConstantIndicesTensorKernelGenerator {
public:
  GatherConstantIndicesTensorKernelGeneratorImpl(Meta &&input_meta,
                                                 Meta &&output_meta,
                                                 Tensor &&indices,
                                                 int64_t axis);
  GatherConstantIndicesTensorKernelGeneratorImpl(
      const GatherConstantIndicesTensorKernelGeneratorImpl &) = delete;
  GatherConstantIndicesTensorKernelGeneratorImpl(
      GatherConstantIndicesTensorKernelGeneratorImpl &&) = default;
  virtual ~GatherConstantIndicesTensorKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<GatherConstantIndicesTensorKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool
  Equals(const GatherConstantIndicesTensorKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Tensor indices_;
  const int64_t axis_;
};

class GatherConstantDataTensorKernelGeneratorImpl
    : public GatherConstantDataTensorKernelGenerator {
public:
  GatherConstantDataTensorKernelGeneratorImpl(Meta &&input_meta,
                                              Meta &&output_meta,
                                              Tensor &&data);
  GatherConstantDataTensorKernelGeneratorImpl(
      const GatherConstantDataTensorKernelGeneratorImpl &) = delete;
  GatherConstantDataTensorKernelGeneratorImpl(
      GatherConstantDataTensorKernelGeneratorImpl &&) = default;
  virtual ~GatherConstantDataTensorKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<GatherConstantDataTensorKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;
  size_t GetHashCode() const override;
  bool Equals(const KernelGenerator &other) const override;
  bool Equals(const GatherConstantDataTensorKernelGeneratorImpl &other) const;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Tensor data_;
};

std::unique_ptr<GatherConstantIndexScalarKernelGenerator>
GatherConstantIndexScalarKernelGenerator::Make(Meta &&input_meta,
                                               Meta &&output_meta, int64_t axis,
                                               int64_t index) {
  return std::make_unique<GatherConstantIndexScalarKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), axis, index);
}

std::unique_ptr<GatherConstantIndicesTensorKernelGenerator>
GatherConstantIndicesTensorKernelGenerator::Make(Meta &&input_meta,
                                                 Meta &&output_meta,
                                                 Tensor &&indices,
                                                 int64_t axis) {
  return std::make_unique<GatherConstantIndicesTensorKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(indices), axis);
}

std::unique_ptr<GatherConstantDataTensorKernelGenerator>
GatherConstantDataTensorKernelGenerator::Make(Meta &&input_meta,
                                              Meta &&output_meta,
                                              Tensor &&data) {
  return std::make_unique<GatherConstantDataTensorKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(data));
}

GatherConstantIndexScalarKernelGeneratorImpl::
    GatherConstantIndexScalarKernelGeneratorImpl(Meta &&input_meta,
                                                 Meta &&output_meta,
                                                 int64_t axis, int64_t index)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      axis_(axis), index_(index) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
GatherConstantIndexScalarKernelGeneratorImpl::
    YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                        llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<GatherConstantIndexScalarKernel>
GatherConstantIndexScalarKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<GatherConstantIndexScalarKernel>(axis_, index_);
}

const Meta &GatherConstantIndexScalarKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &
GatherConstantIndexScalarKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string
GatherConstantIndexScalarKernelGeneratorImpl::GetKernelName() const {
  return GatherConstantIndexScalarKernel::kKernelName;
}

size_t GatherConstantIndexScalarKernelGeneratorImpl::GetHashCode() const {
  size_t hash = GetInputMeta().GetHashCode();
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= axis_ + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= index_ + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool GatherConstantIndexScalarKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const GatherConstantIndexScalarKernelGeneratorImpl *other_ptr =
          dynamic_cast<const GatherConstantIndexScalarKernelGeneratorImpl *>(
              &other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool GatherConstantIndexScalarKernelGeneratorImpl::Equals(
    const GatherConstantIndexScalarKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() && axis_ == other.axis_ &&
         index_ == other.index_;
}

GatherConstantIndicesTensorKernelGeneratorImpl::
    GatherConstantIndicesTensorKernelGeneratorImpl(Meta &&input_meta,
                                                   Meta &&output_meta,
                                                   Tensor &&indices,
                                                   int64_t axis)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      indices_(std::move(indices)), axis_(axis) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
GatherConstantIndicesTensorKernelGeneratorImpl::
    YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                        llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<GatherConstantIndicesTensorKernel>
GatherConstantIndicesTensorKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  Tensor indices = indices_;
  return std::make_shared<GatherConstantIndicesTensorKernel>(std::move(indices),
                                                             axis_);
}

const Meta &
GatherConstantIndicesTensorKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &
GatherConstantIndicesTensorKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string
GatherConstantIndicesTensorKernelGeneratorImpl::GetKernelName() const {
  return GatherConstantIndicesTensorKernel::kKernelName;
}

size_t GatherConstantIndicesTensorKernelGeneratorImpl::GetHashCode() const {
  size_t hash = GetInputMeta().GetHashCode();
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= indices_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= axis_ + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool GatherConstantIndicesTensorKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const GatherConstantIndicesTensorKernelGeneratorImpl *other_ptr =
          dynamic_cast<const GatherConstantIndicesTensorKernelGeneratorImpl *>(
              &other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool GatherConstantIndicesTensorKernelGeneratorImpl::Equals(
    const GatherConstantIndicesTensorKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() &&
         indices_ == other.indices_ && axis_ == other.axis_;
}

GatherConstantDataTensorKernelGeneratorImpl::
    GatherConstantDataTensorKernelGeneratorImpl(Meta &&input_meta,
                                                Meta &&output_meta,
                                                Tensor &&data)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      data_(std::move(data)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
GatherConstantDataTensorKernelGeneratorImpl::
    YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                        llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<GatherConstantDataTensorKernel>
GatherConstantDataTensorKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  Tensor data = data_;
  return std::make_shared<GatherConstantDataTensorKernel>(std::move(data));
}

const Meta &GatherConstantDataTensorKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &GatherConstantDataTensorKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string GatherConstantDataTensorKernelGeneratorImpl::GetKernelName() const {
  return GatherConstantDataTensorKernel::kKernelName;
}

size_t GatherConstantDataTensorKernelGeneratorImpl::GetHashCode() const {
  size_t hash = typeid(GatherConstantDataTensorKernelGeneratorImpl).hash_code();
  hash ^= GetInputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= GetOutputMeta().GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  hash ^= data_.GetHashCode() + kHashSeed + (hash << 6) + (hash >> 2);
  return hash;
}

bool GatherConstantDataTensorKernelGeneratorImpl::Equals(
    const KernelGenerator &other) const {
  if (const GatherConstantDataTensorKernelGeneratorImpl *other_ptr =
          dynamic_cast<const GatherConstantDataTensorKernelGeneratorImpl *>(
              &other)) {
    return Equals(*other_ptr);
  } else {
    return false;
  }
}

bool GatherConstantDataTensorKernelGeneratorImpl::Equals(
    const GatherConstantDataTensorKernelGeneratorImpl &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta() && data_ == other.data_;
}

} // namespace kernel
} // namespace fluidml
