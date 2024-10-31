#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_DIV_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_DIV_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/div.h"
#include "structure/tensor/meta.h"

namespace fluidml {
namespace kernel {

class DivConstantRhsKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~DivConstantRhsKernelGenerator() = default;
  virtual std::shared_ptr<DivConstantRhsKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<SingleInputWithoutBufferKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, Type type, float64_t constant);

protected:
  DivConstantRhsKernelGenerator() = default;
  DivConstantRhsKernelGenerator(const DivConstantRhsKernelGenerator &) = delete;
  DivConstantRhsKernelGenerator(DivConstantRhsKernelGenerator &&) = default;
};

class DivCommonKernelGenerator
    : public DoubleInputsWithoutBufferKernelGenerator {
public:
  virtual ~DivCommonKernelGenerator() = default;
  virtual std::shared_ptr<DivCommonKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<DoubleInputsWithoutBufferKernelGenerator>
  Make(Meta &&lhs_meta, Meta &&rhs_meta, Meta &&output_meta);

protected:
  DivCommonKernelGenerator() = default;
  DivCommonKernelGenerator(const DivCommonKernelGenerator &) = delete;
  DivCommonKernelGenerator(DivCommonKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif