#ifndef CPU_TRANSFORM_KERNEL_KERNEL_SLICE_H_
#define CPU_TRANSFORM_KERNEL_KERNEL_SLICE_H_

#include "structure/kernel/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

class SliceKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "SliceKernel";
  SliceKernel(llvm::SmallVector<llvm::SmallVector<int64_t, 4>> &&informations);
  SliceKernel(const SliceKernel &) = delete;
  SliceKernel(SliceKernel &&) = default;
  virtual ~SliceKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const llvm::SmallVector<llvm::SmallVector<int64_t, 4>> informations_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
