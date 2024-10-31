#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_DIV_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_DIV_H_

#include "structure/kernel/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace fluidml {
namespace kernel {

class DivKernel : virtual public Kernel {
public:
  DivKernel() = default;
  DivKernel(const DivKernel &) = delete;
  DivKernel(DivKernel &&) = default;
  virtual ~DivKernel() = default;
};

class DivConstantRhsKernel : public DivKernel,
                             public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "DivConstantRhsKernel";
  DivConstantRhsKernel(Type type, float64_t constant);
  DivConstantRhsKernel(const DivConstantRhsKernel &) = delete;
  DivConstantRhsKernel(DivConstantRhsKernel &&) = default;
  virtual ~DivConstantRhsKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const Type type_;
  const float64_t constant_;
};

class DivCommonKernel : public DivKernel,
                        public DoubleInputsWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "DivCommonKernel";
  DivCommonKernel() = default;
  DivCommonKernel(const DivCommonKernel &) = delete;
  DivCommonKernel(DivCommonKernel &&) = default;
  virtual ~DivCommonKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace fluidml

#endif