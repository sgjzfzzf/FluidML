#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_WHERE_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_WHERE_H_

#include "structure/kernel/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace fluidml {
namespace kernel {

class WhereConstantCondConstantScalarYKernel
    : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] =
      "WhereConstantCondConstantScalarYKernel";
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
  const Tensor cond_;
  const Type type_;
  const float64_t y_;
};

class WhereConstantCondConstantTensorYKernel
    : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] =
      "WhereConstantCondConstantTensorYKernel";
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
  const Tensor cond_;
  const Tensor y_;
};

} // namespace kernel
} // namespace fluidml

#endif
