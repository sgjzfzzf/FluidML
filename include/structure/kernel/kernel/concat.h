#ifndef CPU_TRANSFORMERS_KERNEL_KERNEL_CONCAT_H_
#define CPU_TRANSFORMERS_KERNEL_KERNEL_CONCAT_H_

#include "structure/kernel/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

class ConcatKernel : virtual public Kernel {
public:
  ConcatKernel(int64_t axis);
  ConcatKernel(const ConcatKernel &) = delete;
  ConcatKernel(ConcatKernel &&) = default;
  virtual ~ConcatKernel() = default;

protected:
  const int64_t axis_;
};

class Concat2Kernel : public ConcatKernel,
                      public DoubleInputsWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "Concat2Kernel";
  Concat2Kernel(int64_t axis);
  Concat2Kernel(const Concat2Kernel &) = delete;
  Concat2Kernel(Concat2Kernel &&) = default;
  virtual ~Concat2Kernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
