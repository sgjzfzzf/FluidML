#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_ADD_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_ADD_H_

#include "mlir/IR/Builders.h"
#include "structure/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class AddKernel : virtual public Kernel {
public:
  AddKernel() = default;
  AddKernel(const AddKernel &add_node) = delete;
  AddKernel(AddKernel &&add_node) = default;
  virtual ~AddKernel() = default;
};

class AddConstantKernel : public AddKernel,
                          public SingleInputWithoutBufferKernel {
public:
  AddConstantKernel(Type type, float64_t constant);
  AddConstantKernel(const AddConstantKernel &add_kernel) = delete;
  AddConstantKernel(AddConstantKernel &&add_kernel) = default;
  virtual ~AddConstantKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  Type type_;
  float64_t constant_;
  static constexpr char kKernelName[] = "AddConstantKernel";
};

class AddCommonKernel : public DoubleInputsWithoutBufferKernel {
public:
  AddCommonKernel() = default;
  AddCommonKernel(const AddCommonKernel &add_kernel) = delete;
  AddCommonKernel(AddCommonKernel &&add_kernel) = default;
  virtual ~AddCommonKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "AddCommonKernel";
};

} // namespace kernel
} // namespace cpu_transformers

#endif
