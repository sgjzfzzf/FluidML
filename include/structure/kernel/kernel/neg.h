#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_KERNEL_NEG_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_KERNEL_NEG_H_

#include "mlir/IR/Builders.h"
#include "structure/kernel/kernel/kernel.h"
#include <string>

namespace fluidml {
namespace kernel {

class NegKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "NegKernel";
  NegKernel() = default;
  NegKernel(const NegKernel &) = delete;
  NegKernel(NegKernel &&) = default;
  virtual ~NegKernel() = default;
  std::string GetKernelName() const;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const;
};

} // namespace kernel
} // namespace fluidml

#endif
