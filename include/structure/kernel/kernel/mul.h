#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_MUL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_MUL_H_

#include "structure/kernel/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class MulKernel : virtual public Kernel {
public:
  MulKernel() = default;
  MulKernel(const MulKernel &other) = delete;
  MulKernel(MulKernel &&other) = default;
  virtual ~MulKernel() = default;
};

class MulConstantKernel : public SingleInputWithoutBufferKernel,
                          public MulKernel {
public:
  static constexpr char kKernelName[] = "MulConstantKernel";
  MulConstantKernel(Type type, float64_t constant);
  MulConstantKernel(const MulConstantKernel &other) = delete;
  MulConstantKernel(MulConstantKernel &&other) = default;
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
  MulCommonKernel(const MulCommonKernel &other) = delete;
  MulCommonKernel(MulCommonKernel &&other) = default;
  virtual ~MulCommonKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
