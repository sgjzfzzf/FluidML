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
  Tensor cond_;
  Type type_;
  float64_t y_;
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
  Tensor cond_;
  Tensor y_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
