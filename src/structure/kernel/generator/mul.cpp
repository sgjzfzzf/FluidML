#include "structure/kernel/generator/mul.h"

namespace cpu_transformers {
namespace kernel {

class MulConstantKernelGeneratorImpl : public MulConstantKernelGenerator {
public:
  MulConstantKernelGeneratorImpl(Type type, float64_t constant);
  MulConstantKernelGeneratorImpl(
      const MulConstantKernelGeneratorImpl &generator) = delete;
  MulConstantKernelGeneratorImpl(MulConstantKernelGeneratorImpl &&generator) =
      default;
  virtual ~MulConstantKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<MulConstantKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const Type type_;
  const float64_t constant_;
};

class MulCommonKernelGeneratorImpl : public MulCommonKernelGenerator {
public:
  MulCommonKernelGeneratorImpl() = default;
  MulCommonKernelGeneratorImpl(const MulCommonKernelGeneratorImpl &generator) =
      delete;
  MulCommonKernelGeneratorImpl(MulCommonKernelGeneratorImpl &&generator) =
      default;
  virtual ~MulCommonKernelGeneratorImpl() = default;
  std::shared_ptr<DoubleInputsWithoutBufferKernel>
  YieldDoubleInputsWithoutBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<MulCommonKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) override;
};

std::unique_ptr<MulConstantKernelGenerator>
MulConstantKernelGenerator::Make(Type type, float64_t constant) {
  return std::make_unique<MulConstantKernelGeneratorImpl>(type, constant);
}

std::unique_ptr<MulCommonKernelGenerator> MulCommonKernelGenerator::Make() {
  return std::make_unique<MulCommonKernelGeneratorImpl>();
}

MulConstantKernelGeneratorImpl::MulConstantKernelGeneratorImpl(
    Type type, float64_t constant)
    : type_(type), constant_(constant) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
MulConstantKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<MulConstantKernel>
MulConstantKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                      llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<MulConstantKernel>(type_, constant_);
}

std::shared_ptr<DoubleInputsWithoutBufferKernel>
MulCommonKernelGeneratorImpl::YieldDoubleInputsWithoutBufferKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return std::make_unique<MulCommonKernel>();
}

std::shared_ptr<MulCommonKernel>
MulCommonKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> lhs_layout,
                                    llvm::ArrayRef<size_t> rhs_layout,
                                    llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<MulCommonKernel>();
}

} // namespace kernel
} // namespace cpu_transformers
