#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_CUM_SUM_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_CUM_SUM_H_

#include "structure/kernel/kernel/kernel.h"

namespace fluidml {
namespace kernel {

class CumSumKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "CumSumKernel";
  CumSumKernel(int64_t axis, bool exclusive, bool reverse);
  CumSumKernel(const CumSumKernel &) = delete;
  CumSumKernel(CumSumKernel &&) = default;
  virtual ~CumSumKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const int64_t axis_;
  const bool exclusive_;
  const bool reverse_;
};

} // namespace kernel
} // namespace fluidml

#endif