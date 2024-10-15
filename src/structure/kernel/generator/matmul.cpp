#include "structure/kernel/generator/matmul.h"

namespace cpu_transformers {
namespace kernel {

class MatMulKernelGeneratorImpl : public MatMulKernelGenerator {
public:
  MatMulKernelGeneratorImpl() = default;
  MatMulKernelGeneratorImpl(const MatMulKernelGeneratorImpl &generator) =
      delete;
  MatMulKernelGeneratorImpl(MatMulKernelGeneratorImpl &&generator) = default;
  virtual ~MatMulKernelGeneratorImpl() = default;
  std::shared_ptr<DoubleInputsWithoutBufferKernel>
  YieldDoubleInputsWithoutBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<MatMulKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) override;
};

std::unique_ptr<MatMulKernelGenerator> MatMulKernelGenerator::Make() {
  return std::make_unique<MatMulKernelGeneratorImpl>();
}

std::shared_ptr<DoubleInputsWithoutBufferKernel>
MatMulKernelGeneratorImpl::YieldDoubleInputsWithoutBufferKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return Yield(lhs_layout, rhs_layout, output_layout);
}

std::shared_ptr<MatMulKernel>
MatMulKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> lhs_layout,
                                 llvm::ArrayRef<size_t> rhs_layout,
                                 llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<MatMulKernel>();
}

} // namespace kernel
} // namespace cpu_transformers
