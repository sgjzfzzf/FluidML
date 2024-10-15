#include "structure/kernel/generator/div.h"

namespace cpu_transformers {
namespace kernel {

class DivConstantRhsKernelGeneratorImpl : public DivConstantRhsKernelGenerator {
public:
  DivConstantRhsKernelGeneratorImpl(Type type, float64_t constant);
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

private:
  const Type type_;
  const float64_t constant_;
};

std::unique_ptr<SingleInputWithoutBufferKernelGenerator>
DivConstantRhsKernelGenerator::Make(Type type, float64_t constant) {
  return std::make_unique<DivConstantRhsKernelGeneratorImpl>(type, constant);
}

DivConstantRhsKernelGeneratorImpl::DivConstantRhsKernelGeneratorImpl(
    Type type, float64_t constant)
    : type_(type), constant_(constant) {}

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

} // namespace kernel
} // namespace cpu_transformers
