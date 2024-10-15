#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_DIV_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_DIV_H_

#include "structure/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class DivConstantRhsKernel : public SingleInputWithoutBufferKernel {
public:
  DivConstantRhsKernel(Type type, float64_t constant);
  DivConstantRhsKernel(const DivConstantRhsKernel &div_kernel) = delete;
  DivConstantRhsKernel(DivConstantRhsKernel &&div_kernel) = default;
  virtual ~DivConstantRhsKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "DivConstantRhsKernel";
  const Type type_;
  const float64_t constant_;
};

class DivConstantRhsKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~DivConstantRhsKernelGenerator() = default;
  virtual std::shared_ptr<DivConstantRhsKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<SingleInputWithoutBufferKernelGenerator>
  Make(Type type, float64_t constant);

protected:
  DivConstantRhsKernelGenerator() = default;
  DivConstantRhsKernelGenerator(
      const DivConstantRhsKernelGenerator &generator) = delete;
  DivConstantRhsKernelGenerator(DivConstantRhsKernelGenerator &&generator) =
      default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif