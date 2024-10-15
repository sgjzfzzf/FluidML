#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_SOFTMAX_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_SOFTMAX_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "structure/kernel/kernel.h"
#include <cstdint>

namespace cpu_transformers {
namespace kernel {

class SoftmaxKernel : public SingleInputWithBufferKernel {
public:
  SoftmaxKernel(int64_t axis);
  SoftmaxKernel(const SoftmaxKernel &) = delete;
  SoftmaxKernel(SoftmaxKernel &&) = default;
  virtual ~SoftmaxKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input, mlir::Value &output,
           mlir::Value &buffer) const override;

private:
  static constexpr char kKernelName[] = "SoftmaxKernel";
  const int64_t axis_;
};

class SoftmaxKernelGenerator : public SingleInputWithBufferKernelGenerator {
public:
  virtual ~SoftmaxKernelGenerator() = default;
  virtual std::shared_ptr<SoftmaxKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<SoftmaxKernelGenerator> Make(int64_t axis);

protected:
  SoftmaxKernelGenerator() = default;
  SoftmaxKernelGenerator(const SoftmaxKernelGenerator &) = delete;
  SoftmaxKernelGenerator(SoftmaxKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
