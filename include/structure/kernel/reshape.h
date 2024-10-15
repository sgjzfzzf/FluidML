#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_RESHAPE_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_RESHAPE_H_

#include "structure/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

class ReshapeKernel : public SingleInputWithoutBufferKernel {
public:
  ReshapeKernel() = default;
  ReshapeKernel(const ReshapeKernel &) = delete;
  ReshapeKernel(ReshapeKernel &&) = default;
  virtual ~ReshapeKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "ReshapeKernel";
};

class ReshapeKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~ReshapeKernelGenerator() = default;
  virtual std::shared_ptr<ReshapeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<ReshapeKernelGenerator> Make();

protected:
  ReshapeKernelGenerator() = default;
  ReshapeKernelGenerator(const ReshapeKernelGenerator &) = delete;
  ReshapeKernelGenerator(ReshapeKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
