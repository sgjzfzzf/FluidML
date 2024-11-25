#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_MUL_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_MUL_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/mul.h"
#include "structure/tensor/meta.h"

namespace fluidml {
namespace kernel {

class MulConstantKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~MulConstantKernelGenerator() = default;
  virtual std::shared_ptr<MulConstantKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<MulConstantKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, Type type, float64_t constant);

protected:
  MulConstantKernelGenerator() = default;
  MulConstantKernelGenerator(const MulConstantKernelGenerator &) = delete;
  MulConstantKernelGenerator(MulConstantKernelGenerator &&) = default;
};

class MulCommonKernelGenerator
    : public DoubleInputsWithoutBufferKernelGenerator {
public:
  virtual ~MulCommonKernelGenerator() = default;
  virtual std::shared_ptr<MulCommonKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<MulCommonKernelGenerator>
  Make(Meta &&lhs_meta, Meta &&rhs_meta, Meta &&output_meta);

protected:
  MulCommonKernelGenerator() = default;
  MulCommonKernelGenerator(const MulCommonKernelGenerator &) = delete;
  MulCommonKernelGenerator(MulCommonKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
