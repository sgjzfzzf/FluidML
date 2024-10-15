#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_SPLIT_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_SPLIT_H_

#include "structure/kernel/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

// TODO: need further implementation for split
class SplitKernel : public Kernel {
public:
  SplitKernel(int64_t axis);
  SplitKernel(const SplitKernel &other) = delete;
  SplitKernel(SplitKernel &&other) = default;
  virtual ~SplitKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::ValueRange outputs);

private:
  static constexpr char kKernelName[] = "SplitKernel";
  int64_t axis_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
