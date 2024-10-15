#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_ERF_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_ERF_H_

#include "structure/kernel/kernel.h"
#include <memory>

namespace cpu_transformers {
namespace kernel {

class ErfKernel : public SingleInputWithoutBufferKernel {
public:
  ErfKernel() = default;
  ErfKernel(const ErfKernel &erf_kernel) = delete;
  ErfKernel(ErfKernel &&erf_kernel) = default;
  virtual ~ErfKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "ErfKernel";
};

class ErfKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~ErfKernelGenerator() = default;
  virtual std::shared_ptr<ErfKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<ErfKernelGenerator> Make();

protected:
  ErfKernelGenerator() = default;
  ErfKernelGenerator(const ErfKernelGenerator &generator) = delete;
  ErfKernelGenerator(ErfKernelGenerator &&generator) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif