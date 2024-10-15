#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_UNSQUEEZE_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_UNSQUEEZE_H_

#include "structure/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

class UnSqueezeKernel : public SingleInputWithoutBufferKernel {
public:
  UnSqueezeKernel(std::vector<int64_t> &&axes);
  UnSqueezeKernel(const UnSqueezeKernel &other) = delete;
  UnSqueezeKernel(UnSqueezeKernel &&other) = default;
  virtual ~UnSqueezeKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "UnSqueezeKernel";
  const std::vector<int64_t> axes_;
};

class UnSqueezeKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~UnSqueezeKernelGenerator() = default;
  virtual std::shared_ptr<UnSqueezeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<UnSqueezeKernelGenerator>
  Make(std::vector<int64_t> axes);

protected:
  UnSqueezeKernelGenerator() = default;
  UnSqueezeKernelGenerator(const UnSqueezeKernelGenerator &generator) = delete;
  UnSqueezeKernelGenerator(UnSqueezeKernelGenerator &&generator) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
