#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_TRANSPOSE_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_TRANSPOSE_H_

#include "structure/kernel/kernel/kernel.h"
#include <cstdint>

namespace fluidml {
namespace kernel {

class TransposeKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "TransposeKernel";
  TransposeKernel(std::vector<int64_t> &&perms);
  TransposeKernel(const TransposeKernel &) = delete;
  TransposeKernel(TransposeKernel &&) = default;
  virtual ~TransposeKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const std::vector<int64_t> perms_;
};

} // namespace kernel
} // namespace fluidml

#endif
