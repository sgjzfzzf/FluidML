#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_UNSQUEEZE_SUB_MUL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_UNSQUEEZE_SUB_MUL_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/unsqueeze_sub_mul.h"

namespace cpu_transformers {
namespace kernel {

class UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator() = default;
  virtual std::shared_ptr<UnsqueezeSubLhsScalarMulRhsScalarKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator>
  Make(std::vector<int64_t> &&unsqueeze_axes, const Type &sub_type,
       float64_t sub_val, const Type &mul_type, float64_t mul_val);

protected:
  UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator() = default;
  UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator(
      const UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator &generator) =
      delete;
  UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator(
      UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator &&generator) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
