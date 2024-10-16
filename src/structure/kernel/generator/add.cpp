#include "structure/kernel/generator/add.h"

namespace cpu_transformers {
namespace kernel {

class AddConstantKernelGeneratorImpl : public AddConstantKernelGenerator {
public:
  AddConstantKernelGeneratorImpl(Type type, float64_t constant);
  AddConstantKernelGeneratorImpl(
      const AddConstantKernelGeneratorImpl &generator) = delete;
  AddConstantKernelGeneratorImpl(AddConstantKernelGeneratorImpl &&generator) =
      default;
  virtual ~AddConstantKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<AddConstantKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  std::string GetKernelName() const override;

private:
  const Type type_;
  const float64_t constant_;
};

class AddCommonKernelGeneratorImpl : public AddCommonKernelGenerator {
public:
  AddCommonKernelGeneratorImpl() = default;
  AddCommonKernelGeneratorImpl(const AddCommonKernelGeneratorImpl &generator) =
      delete;
  AddCommonKernelGeneratorImpl(AddCommonKernelGeneratorImpl &&generator) =
      default;
  virtual ~AddCommonKernelGeneratorImpl() = default;
  std::shared_ptr<DoubleInputsWithoutBufferKernel>
  YieldDoubleInputsWithoutBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<AddCommonKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) override;
  std::string GetKernelName() const override;
};

std::unique_ptr<AddConstantKernelGenerator>
AddConstantKernelGenerator::Make(Type type, float64_t constant) {
  return std::make_unique<AddConstantKernelGeneratorImpl>(type, constant);
}

std::unique_ptr<AddCommonKernelGenerator> AddCommonKernelGenerator::Make() {
  return std::make_unique<AddCommonKernelGeneratorImpl>();
}

AddConstantKernelGeneratorImpl::AddConstantKernelGeneratorImpl(
    Type type, float64_t constant)
    : type_(type), constant_(constant) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
AddConstantKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<AddConstantKernel>
AddConstantKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                      llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<AddConstantKernel>(type_, constant_);
}

std::string AddConstantKernelGeneratorImpl::GetKernelName() const {
  return AddConstantKernel::kKernelName;
}

std::shared_ptr<DoubleInputsWithoutBufferKernel>
AddCommonKernelGeneratorImpl::YieldDoubleInputsWithoutBufferKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return Yield(lhs_layout, rhs_layout, output_layout);
}

std::shared_ptr<AddCommonKernel>
AddCommonKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> lhs_layout,
                                    llvm::ArrayRef<size_t> rhs_layout,
                                    llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<AddCommonKernel>();
}

std::string AddCommonKernelGeneratorImpl::GetKernelName() const {
  return AddCommonKernel::kKernelName;
}

} // namespace kernel
} // namespace cpu_transformers
