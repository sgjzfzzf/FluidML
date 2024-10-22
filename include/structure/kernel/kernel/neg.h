#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_NEG_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_NEG_H_

#include "mlir/IR/Builders.h"
#include "structure/kernel/kernel/kernel.h"
#include <string>

namespace cpu_transformers {
namespace kernel {

class NegKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "NegKernel";
  NegKernel() = default;
  NegKernel(const NegKernel &other) = delete;
  NegKernel(NegKernel &&other) = default;
  virtual ~NegKernel() = default;
  std::string GetKernelName() const;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
