#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_EQUAL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_EQUAL_H_

#include "structure/kernel/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

class EqualKernel : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "EqualKernel";
  EqualKernel(Type type, float64_t value);
  EqualKernel(const EqualKernel &) = delete;
  EqualKernel(EqualKernel &&) = default;
  virtual ~EqualKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  const Type type_;
  const float64_t value_;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
