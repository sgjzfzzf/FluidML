#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_SUB_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_SUB_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/sub.h"
#include "structure/tensor/meta.h"

namespace fluidml {
namespace kernel {

class SubConstantLhsKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~SubConstantLhsKernelGenerator() = default;
  virtual std::shared_ptr<SubConstantLhsKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<SubConstantLhsKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, Type type, float64_t value);

protected:
  SubConstantLhsKernelGenerator() = default;
  SubConstantLhsKernelGenerator(const SubConstantLhsKernelGenerator &) = delete;
  SubConstantLhsKernelGenerator(SubConstantLhsKernelGenerator &&) = default;
};

class SubCommonKernelGenerator
    : public DoubleInputsWithoutBufferKernelGenerator {
public:
  virtual ~SubCommonKernelGenerator() = default;
  virtual std::shared_ptr<SubCommonKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<SubCommonKernelGenerator>
  Make(Meta &&lhs_meta, Meta &&rhs_meta, Meta &&output_meta);

protected:
  SubCommonKernelGenerator() = default;
  SubCommonKernelGenerator(const SubCommonKernelGenerator &) = delete;
  SubCommonKernelGenerator(SubCommonKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
