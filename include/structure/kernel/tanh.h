#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_TANH_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_TANH_H_

#include "structure/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

class TanhKernel : public SingleInputWithoutBufferKernel {
public:
  TanhKernel() = default;
  TanhKernel(const TanhKernel &other) = delete;
  TanhKernel(TanhKernel &&other) = default;
  virtual ~TanhKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "TanhKernel";
};

class TanhKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~TanhKernelGenerator() = default;
  virtual std::shared_ptr<TanhKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<TanhKernelGenerator> Make();

protected:
  TanhKernelGenerator() = default;
  TanhKernelGenerator(const TanhKernelGenerator &) = delete;
  TanhKernelGenerator(TanhKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
