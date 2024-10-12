#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_TRANSPOSE_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_TRANSPOSE_H_

#include "structure/kernel/kernel.h"
#include <cstdint>

namespace cpu_transformers {
namespace kernel {

class TransposeKernel : public SingleInputWithoutBufferKernel {
public:
  TransposeKernel(std::vector<int64_t> perms);
  TransposeKernel(const TransposeKernel &) = delete;
  TransposeKernel(TransposeKernel &&) = default;
  virtual ~TransposeKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "TransposeKernel";
  std::vector<int64_t> perms_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
