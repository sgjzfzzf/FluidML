#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_CAST_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_CAST_H_

#include "structure/kernel/kernel/kernel.h"

namespace cpu_transformers {
namespace kernel {

class CastKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "CastKernel";
  CastKernel() = default;
  CastKernel(const CastKernel &) = delete;
  CastKernel(CastKernel &&) = default;
  virtual ~CastKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
