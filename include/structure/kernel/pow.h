#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_POW_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_POW_H_

#include "structure/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class PowKernel : public SingleInputWithoutBufferKernel {
public:
  PowKernel(Type type, float64_t exp);
  PowKernel(const PowKernel &other) = delete;
  PowKernel(PowKernel &&other) = default;
  virtual ~PowKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  static constexpr char kKernelName[] = "PowKernel";
  Type type_;
  float64_t exp_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
