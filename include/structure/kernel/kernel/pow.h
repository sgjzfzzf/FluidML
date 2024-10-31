#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_POW_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_POW_H_

#include "structure/kernel/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace fluidml {
namespace kernel {

class PowKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "PowKernel";
  PowKernel(Type type, float64_t exp);
  PowKernel(const PowKernel &) = delete;
  PowKernel(PowKernel &&) = default;
  virtual ~PowKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const Type type_;
  const float64_t exp_;
};

} // namespace kernel
} // namespace fluidml

#endif
