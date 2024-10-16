#include "structure/kernel/generator/matmul.h"

namespace cpu_transformers {
namespace kernel {

class MatMulKernelGeneratorImpl : public MatMulKernelGenerator {
public:
  MatMulKernelGeneratorImpl(Meta &&lhs_meta, Meta &&rhs_meta,
                            Meta &&output_meta);
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
  const Meta &GetLhsMeta() const override;
  const Meta &GetRhsMeta() const override;
  const Meta &GetOutputMeta() const override;
  std::string GetKernelName() const override;

private:
  const Meta lhs_meta_;
  const Meta rhs_meta_;
  const Meta output_meta_;
};

std::unique_ptr<MatMulKernelGenerator>
MatMulKernelGenerator::Make(Meta &&lhs_meta, Meta &&rhs_meta,
                            Meta &&output_meta) {
  return std::make_unique<MatMulKernelGeneratorImpl>(
      std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta));
}

MatMulKernelGeneratorImpl::MatMulKernelGeneratorImpl(Meta &&lhs_meta,
                                                     Meta &&rhs_meta,
                                                     Meta &&output_meta)
    : lhs_meta_(std::move(lhs_meta)), rhs_meta_(std::move(rhs_meta)),
      output_meta_(std::move(output_meta)) {}

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

const Meta &MatMulKernelGeneratorImpl::GetLhsMeta() const { return lhs_meta_; }

const Meta &MatMulKernelGeneratorImpl::GetRhsMeta() const { return rhs_meta_; }

const Meta &MatMulKernelGeneratorImpl::GetOutputMeta() const {
  return output_meta_;
}

std::string MatMulKernelGeneratorImpl::GetKernelName() const {
  return MatMulKernel::kKernelName;
}

} // namespace kernel
} // namespace cpu_transformers
