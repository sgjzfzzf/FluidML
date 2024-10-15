#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_WHERE_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_WHERE_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace cpu_transformers {
namespace kernel {

class WhereConstantCondConstantScalarYKernel
    : public SingleInputWithoutBufferKernel {
public:
  WhereConstantCondConstantScalarYKernel(Tensor &&cond, Type type, float64_t y);
  WhereConstantCondConstantScalarYKernel(
      const WhereConstantCondConstantScalarYKernel &) = delete;
  WhereConstantCondConstantScalarYKernel(
      WhereConstantCondConstantScalarYKernel &&) = default;
  virtual ~WhereConstantCondConstantScalarYKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] =
      "WhereConstantCondConstantScalarYKernel";
  const Tensor cond_;
  const Type type_;
  const float64_t y_;
};

class WhereConstantCondConstantTensorYKernel
    : public SingleInputWithoutBufferKernel {
public:
  WhereConstantCondConstantTensorYKernel(Tensor &&cond, Tensor &&y);
  WhereConstantCondConstantTensorYKernel(
      const WhereConstantCondConstantTensorYKernel &) = delete;
  WhereConstantCondConstantTensorYKernel(
      WhereConstantCondConstantTensorYKernel &&) = default;
  virtual ~WhereConstantCondConstantTensorYKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] =
      "WhereConstantCondConstantTensorYKernel";
  const Tensor cond_;
  const Tensor y_;
};

class WhereConstantCondConstantScalarYKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~WhereConstantCondConstantScalarYKernelGenerator() = default;
  virtual std::shared_ptr<WhereConstantCondConstantScalarYKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<WhereConstantCondConstantScalarYKernelGenerator>
  Make(Tensor &&cond, Type type, float64_t y);

protected:
  WhereConstantCondConstantScalarYKernelGenerator() = default;
  WhereConstantCondConstantScalarYKernelGenerator(
      const WhereConstantCondConstantScalarYKernelGenerator &) = delete;
  WhereConstantCondConstantScalarYKernelGenerator(
      WhereConstantCondConstantScalarYKernelGenerator &&) = default;
};

class WhereConstantCondConstantTensorYKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~WhereConstantCondConstantTensorYKernelGenerator() = default;
  virtual std::shared_ptr<WhereConstantCondConstantTensorYKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<WhereConstantCondConstantTensorYKernelGenerator>
  Make(Tensor &&cond, Tensor &&y);

protected:
  WhereConstantCondConstantTensorYKernelGenerator() = default;
  WhereConstantCondConstantTensorYKernelGenerator(
      const WhereConstantCondConstantTensorYKernelGenerator &) = delete;
  WhereConstantCondConstantTensorYKernelGenerator(
      WhereConstantCondConstantTensorYKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
