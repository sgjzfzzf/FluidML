#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_UNSQUEEZE_SUB_MUL_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_UNSQUEEZE_SUB_MUL_H_

#include "structure/kernel/kernel/kernel.h"
#include "utils/float.h"
#include "utils/type.h"

namespace fluidml {
namespace kernel {

class UnsqueezeSubLhsScalarMulRhsScalarKernel
    : public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] =
      "UnsqueezeSubLhsScalarMulRhsScalarKernel";
  UnsqueezeSubLhsScalarMulRhsScalarKernel(std::vector<int64_t> &&unsqueeze_axes,
                                          const Type &sub_type,
                                          float64_t sub_val,
                                          const Type &mul_type,
                                          float64_t mul_val);
  UnsqueezeSubLhsScalarMulRhsScalarKernel(
      const UnsqueezeSubLhsScalarMulRhsScalarKernel &) = delete;
  UnsqueezeSubLhsScalarMulRhsScalarKernel(
      UnsqueezeSubLhsScalarMulRhsScalarKernel &&) = default;
  virtual ~UnsqueezeSubLhsScalarMulRhsScalarKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;

private:
  std::vector<int64_t> unsqueeze_axes_;
  const Type sub_type_;
  const float64_t sub_val_;
  const Type mul_type_;
  const float64_t mul_val_;
};

} // namespace kernel
} // namespace fluidml

#endif
