#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_MUL_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_MUL_H_

#include "structure/kernel/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace fluidml {
namespace kernel {

class MulKernel : virtual public Kernel {
public:
  MulKernel() = default;
  MulKernel(const MulKernel &) = delete;
  MulKernel(MulKernel &&) = default;
  virtual ~MulKernel() = default;
};

class MulConstantKernel : public SingleInputWithoutBufferKernel,
                          public MulKernel {
public:
  static constexpr char kKernelName[] = "MulConstantKernel";
  MulConstantKernel(Type type, float64_t constant);
  MulConstantKernel(const MulConstantKernel &) = delete;
  MulConstantKernel(MulConstantKernel &&) = default;
  virtual ~MulConstantKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs,
           mlir::Value &output) const override;

private:
  const Type type_;
  const float64_t constant_;
};

class MulCommonKernel : public DoubleInputsWithoutBufferKernel,
                        public MulKernel {
public:
  static constexpr char kKernelName[] = "MulCommonKernel";
  MulCommonKernel() = default;
  MulCommonKernel(const MulCommonKernel &) = delete;
  MulCommonKernel(MulCommonKernel &&) = default;
  virtual ~MulCommonKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace fluidml

#endif
