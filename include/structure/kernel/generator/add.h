#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_ADD_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_ADD_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/add.h"
#include "structure/tensor/meta.h"

namespace cpu_transformers {
namespace kernel {

class AddConstantKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~AddConstantKernelGenerator() = default;
  virtual std::shared_ptr<AddConstantKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<AddConstantKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, Type type, float64_t constant);

protected:
  AddConstantKernelGenerator() = default;
  AddConstantKernelGenerator(const AddConstantKernelGenerator &) = delete;
  AddConstantKernelGenerator(AddConstantKernelGenerator &&) = default;
};

class AddCommonKernelGenerator
    : public DoubleInputsWithoutBufferKernelGenerator {
public:
  virtual ~AddCommonKernelGenerator() = default;
  virtual std::shared_ptr<AddCommonKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<AddCommonKernelGenerator>
  Make(Meta &&lhs_meta, Meta &&rhs_meta, Meta &&output_meta);

protected:
  AddCommonKernelGenerator() = default;
  AddCommonKernelGenerator(const AddCommonKernelGenerator &) = delete;
  AddCommonKernelGenerator(AddCommonKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif