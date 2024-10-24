#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_REDUCE_MEAN_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_REDUCE_MEAN_H_

#include "structure/kernel/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

class ReduceMeanKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "ReduceMeanKernel";
  ReduceMeanKernel(llvm::SmallVector<int64_t> &&axes, bool keep_dims);
  ReduceMeanKernel(const ReduceMeanKernel &) = delete;
  ReduceMeanKernel(ReduceMeanKernel &&) = default;
  virtual ~ReduceMeanKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const llvm::SmallVector<int64_t> axes_;
  const bool keep_dims_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
