#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_MATMUL_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_MATMUL_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/matmul.h"

namespace fluidml {
namespace kernel {

class MatMulKernelGenerator : public DoubleInputsWithoutBufferKernelGenerator {
public:
  virtual ~MatMulKernelGenerator() = default;
  virtual std::shared_ptr<MatMulKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<MatMulKernelGenerator>
  Make(Meta &&lhs_meta, Meta &&rhs_meta, Meta &&output_meta);

protected:
  MatMulKernelGenerator() = default;
  MatMulKernelGenerator(const MatMulKernelGenerator &) = delete;
  MatMulKernelGenerator(MatMulKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
