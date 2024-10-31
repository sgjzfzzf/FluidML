#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_WHERE_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_WHERE_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/where.h"

namespace fluidml {
namespace kernel {

class WhereConstantCondConstantScalarYKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~WhereConstantCondConstantScalarYKernelGenerator() = default;
  virtual std::shared_ptr<WhereConstantCondConstantScalarYKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<WhereConstantCondConstantScalarYKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, Tensor &&cond, Type type,
       float64_t y);

protected:
  WhereConstantCondConstantScalarYKernelGenerator() = default;
  WhereConstantCondConstantScalarYKernelGenerator(
      const WhereConstantCondConstantScalarYKernelGenerator &) = delete;
  WhereConstantCondConstantScalarYKernelGenerator(
      WhereConstantCondConstantScalarYKernelGenerator &&) = default;
};

class WhereConstantCondConstantTensorYKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~WhereConstantCondConstantTensorYKernelGenerator() = default;
  virtual std::shared_ptr<WhereConstantCondConstantTensorYKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<WhereConstantCondConstantTensorYKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, Tensor &&cond, Tensor &&y);

protected:
  WhereConstantCondConstantTensorYKernelGenerator() = default;
  WhereConstantCondConstantTensorYKernelGenerator(
      const WhereConstantCondConstantTensorYKernelGenerator &) = delete;
  WhereConstantCondConstantTensorYKernelGenerator(
      WhereConstantCondConstantTensorYKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
