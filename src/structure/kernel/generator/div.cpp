#include "structure/kernel/generator/div.h"

namespace cpu_transformers {
namespace kernel {

class DivConstantRhsKernelGeneratorImpl : public DivConstantRhsKernelGenerator {
public:
  DivConstantRhsKernelGeneratorImpl(Meta &&input_meta, Meta &&output_meta,
                                    Type type, float64_t constant);
  DivConstantRhsKernelGeneratorImpl(
      const DivConstantRhsKernelGeneratorImpl &generator) = delete;
  DivConstantRhsKernelGeneratorImpl(
      DivConstantRhsKernelGeneratorImpl &&generator) = default;
  virtual ~DivConstantRhsKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<DivConstantRhsKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  const Meta &GetInputMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;

private:
  const Meta input_meta_;
  const Meta output_meta_;
  const Type type_;
  const float64_t constant_;
};

std::unique_ptr<SingleInputWithoutBufferKernelGenerator>
DivConstantRhsKernelGenerator::Make(Meta &&input_meta, Meta &&output_meta,
                                    Type type, float64_t constant) {
  return std::make_unique<DivConstantRhsKernelGeneratorImpl>(
      std::move(input_meta), std::move(output_meta), type, constant);
}

DivConstantRhsKernelGeneratorImpl::DivConstantRhsKernelGeneratorImpl(
    Meta &&input_meta, Meta &&output_meta, Type type, float64_t constant)
    : input_meta_(std::move(input_meta)), output_meta_(std::move(output_meta)),
      type_(type), constant_(constant) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
DivConstantRhsKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<DivConstantRhsKernel>
DivConstantRhsKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                         llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<DivConstantRhsKernel>(type_, constant_);
}

const Meta &DivConstantRhsKernelGeneratorImpl::GetInputMeta() const {
  return input_meta_;
}

const Meta &DivConstantRhsKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string DivConstantRhsKernelGeneratorImpl::GetKernelName() const {
  return DivConstantRhsKernel::kKernelName;
}

} // namespace kernel
} // namespace cpu_transformers
