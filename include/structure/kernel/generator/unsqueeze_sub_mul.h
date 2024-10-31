#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_UNSQUEEZE_SUB_MUL_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_UNSQUEEZE_SUB_MUL_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/unsqueeze_sub_mul.h"
#include "structure/tensor/meta.h"

namespace fluidml {
namespace kernel {

class UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator() = default;
  virtual std::shared_ptr<UnsqueezeSubLhsScalarMulRhsScalarKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta,
       std::vector<int64_t> &&unsqueeze_axes, const Type &sub_type,
       float64_t sub_val, const Type &mul_type, float64_t mul_val);

protected:
  UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator() = default;
  UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator(
      const UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator &) = delete;
  UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator(
      UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
