#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_POW_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_POW_H_

#include "structure/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class PowKernel : public SingleInputWithoutBufferKernel {
public:
  PowKernel(Type type, float64_t exp);
  PowKernel(const PowKernel &other) = delete;
  PowKernel(PowKernel &&other) = default;
  virtual ~PowKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "PowKernel";
  const Type type_;
  const float64_t exp_;
};

class PowKernelGenerator : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~PowKernelGenerator() = default;
  virtual std::shared_ptr<PowKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<PowKernelGenerator> Make(Type type, float64_t exp);

protected:
  PowKernelGenerator() = default;
  PowKernelGenerator(const PowKernelGenerator &generator) = delete;
  PowKernelGenerator(PowKernelGenerator &&generator) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
