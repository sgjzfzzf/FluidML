#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_ADD_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_ADD_H_

#include "mlir/IR/Builders.h"
#include "structure/kernel/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class AddKernel : virtual public Kernel {
public:
  AddKernel() = default;
  AddKernel(const AddKernel &) = delete;
  AddKernel(AddKernel &&) = default;
  virtual ~AddKernel() = default;
};

class AddConstantKernel : public AddKernel,
                          public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "AddConstantKernel";
  AddConstantKernel(Type type, float64_t constant);
  AddConstantKernel(const AddConstantKernel &) = delete;
  AddConstantKernel(AddConstantKernel &&) = default;
  virtual ~AddConstantKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const Type type_;
  const float64_t constant_;
};

class AddCommonKernel : public DoubleInputsWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "AddCommonKernel";
  AddCommonKernel() = default;
  AddCommonKernel(const AddCommonKernel &) = delete;
  AddCommonKernel(AddCommonKernel &&) = default;
  virtual ~AddCommonKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
