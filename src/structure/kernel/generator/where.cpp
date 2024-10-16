#include "structure/kernel/generator/where.h"

namespace cpu_transformers {
namespace kernel {

class WhereConstantCondConstantScalarYKernelGeneratorImpl
    : public WhereConstantCondConstantScalarYKernelGenerator {
public:
  WhereConstantCondConstantScalarYKernelGeneratorImpl(Meta &&input_meta,
                                                      Meta &&output_meta,
                                                      Tensor &&cond, Type type,
                                                      float64_t y);
  WhereConstantCondConstantScalarYKernelGeneratorImpl(
      const WhereConstantCondConstantScalarYKernelGeneratorImpl &generator) =
      delete;
  WhereConstantCondConstantScalarYKernelGeneratorImpl(
      WhereConstantCondConstantScalarYKernelGeneratorImpl &&generator) =
      default;
  virtual ~WhereConstantCondConstantScalarYKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<WhereConstantCondConstantScalarYKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Tensor cond_;
  const Type type_;
  const float64_t y_;
};

class WhereConstantCondConstantTensorYKernelGeneratorImpl
    : public WhereConstantCondConstantTensorYKernelGenerator {
public:
  WhereConstantCondConstantTensorYKernelGeneratorImpl(Meta &&input_meta,
                                                      Meta &&output_meta,
                                                      Tensor &&cond,
                                                      Tensor &&y);
  WhereConstantCondConstantTensorYKernelGeneratorImpl(
      const WhereConstantCondConstantTensorYKernelGeneratorImpl &generator) =
      delete;
  WhereConstantCondConstantTensorYKernelGeneratorImpl(
      WhereConstantCondConstantTensorYKernelGeneratorImpl &&generator) =
      default;
  virtual ~WhereConstantCondConstantTensorYKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<WhereConstantCondConstantTensorYKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Tensor cond_;
  const Tensor y_;
};

std::unique_ptr<WhereConstantCondConstantScalarYKernelGenerator>
WhereConstantCondConstantScalarYKernelGenerator::Make(Meta &&input_meta,
                                                      Meta &&output_meta,
                                                      Tensor &&cond, Type type,
                                                      float64_t y) {
  return std::make_unique<WhereConstantCondConstantScalarYKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(cond), type, y);
}

std::unique_ptr<WhereConstantCondConstantTensorYKernelGenerator>
WhereConstantCondConstantTensorYKernelGenerator::Make(Meta &&input_meta,
                                                      Meta &&output_meta,
                                                      Tensor &&cond,
                                                      Tensor &&y) {
  return std::make_unique<WhereConstantCondConstantTensorYKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), std::move(cond),
      std::move(y));
}

WhereConstantCondConstantScalarYKernelGeneratorImpl::
    WhereConstantCondConstantScalarYKernelGeneratorImpl(Meta &&input_meta,
                                                        Meta &&output_meta,
                                                        Tensor &&cond,
                                                        Type type, float64_t y)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      cond_(std::move(cond)), type_(type), y_(y) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
WhereConstantCondConstantScalarYKernelGeneratorImpl::
    YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                        llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

const Meta &
WhereConstantCondConstantScalarYKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &
WhereConstantCondConstantScalarYKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string
WhereConstantCondConstantScalarYKernelGeneratorImpl::GetKernelName() const {
  return WhereConstantCondConstantScalarYKernel::kKernelName;
}

std::shared_ptr<WhereConstantCondConstantScalarYKernel>
WhereConstantCondConstantScalarYKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  Tensor cond = cond_;
  return std::make_shared<WhereConstantCondConstantScalarYKernel>(
      std::move(cond), type_, y_);
}

WhereConstantCondConstantTensorYKernelGeneratorImpl::
    WhereConstantCondConstantTensorYKernelGeneratorImpl(Meta &&input_meta,
                                                        Meta &&output_meta,
                                                        Tensor &&cond,
                                                        Tensor &&y)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      cond_(std::move(cond)), y_(std::move(y)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
WhereConstantCondConstantTensorYKernelGeneratorImpl::
    YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                        llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<WhereConstantCondConstantTensorYKernel>
WhereConstantCondConstantTensorYKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  Tensor cond = cond_, y = y_;
  return std::make_shared<WhereConstantCondConstantTensorYKernel>(
      std::move(cond), std::move(y));
}

const Meta &
WhereConstantCondConstantTensorYKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &
WhereConstantCondConstantTensorYKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string
WhereConstantCondConstantTensorYKernelGeneratorImpl::GetKernelName() const {
  return WhereConstantCondConstantTensorYKernel::kKernelName;
}

} // namespace kernel
} // namespace cpu_transformers