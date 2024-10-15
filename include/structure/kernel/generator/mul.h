#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_MUL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_MUL_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/mul.h"

namespace cpu_transformers {
namespace kernel {

class MulConstantKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~MulConstantKernelGenerator() = default;
  virtual std::shared_ptr<MulConstantKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<MulConstantKernelGenerator> Make(Type type,
                                                          float64_t constant);

protected:
  MulConstantKernelGenerator() = default;
  MulConstantKernelGenerator(const MulConstantKernelGenerator &generator) =
      delete;
  MulConstantKernelGenerator(MulConstantKernelGenerator &&generator) = default;
};

class MulCommonKernelGenerator
    : public DoubleInputsWithoutBufferKernelGenerator {
public:
  virtual ~MulCommonKernelGenerator() = default;
  virtual std::shared_ptr<MulCommonKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<MulCommonKernelGenerator> Make();

protected:
  MulCommonKernelGenerator() = default;
  MulCommonKernelGenerator(const MulCommonKernelGenerator &generator) = delete;
  MulCommonKernelGenerator(MulCommonKernelGenerator &&generator) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
