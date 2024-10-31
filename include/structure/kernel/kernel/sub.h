#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_SUB_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_SUB_H_

#include "structure/kernel/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace fluidml {
namespace kernel {

class SubKernel : virtual public Kernel {
public:
  SubKernel() = default;
  SubKernel(const SubKernel &sub_kernel) = delete;
  SubKernel(SubKernel &&sub_kernel) = default;
  virtual ~SubKernel() = default;
};

class SubConstantLhsKernel : public SubKernel,
                             public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "SubConstantLhsKernel";
  SubConstantLhsKernel(Type type, float64_t value);
  SubConstantLhsKernel(const SubConstantLhsKernel &) = delete;
  SubConstantLhsKernel(SubConstantLhsKernel &&) = default;
  virtual ~SubConstantLhsKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const Type type_;
  const float64_t value_;
};

class SubCommonKernel : public SubKernel,
                        public DoubleInputsWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "SubCommonKernel";
  SubCommonKernel() = default;
  SubCommonKernel(const SubCommonKernel &sub_kernel) = delete;
  SubCommonKernel(SubCommonKernel &&sub_kernel) = default;
  virtual ~SubCommonKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace fluidml

#endif
