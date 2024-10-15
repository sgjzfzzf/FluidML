#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_TRANSPOSE_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_TRANSPOSE_H_

#include "structure/kernel/kernel.h"
#include <cstdint>

namespace cpu_transformers {
namespace kernel {

class TransposeKernel : public SingleInputWithoutBufferKernel {
public:
  TransposeKernel(std::vector<int64_t> &&perms);
  TransposeKernel(const TransposeKernel &) = delete;
  TransposeKernel(TransposeKernel &&) = default;
  virtual ~TransposeKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "TransposeKernel";
  const std::vector<int64_t> perms_;
};

class TransposeKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~TransposeKernelGenerator() = default;
  virtual std::shared_ptr<TransposeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<TransposeKernelGenerator>
  Make(std::vector<int64_t> perms);

protected:
  TransposeKernelGenerator() = default;
  TransposeKernelGenerator(const TransposeKernelGenerator &) = delete;
  TransposeKernelGenerator(TransposeKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
