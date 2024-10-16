#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_TRANSPOSE_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_GENERATOR_TRANSPOSE_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/transpose.h"
#include "structure/tensor/meta.h"

namespace cpu_transformers {
namespace kernel {

class TransposeKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~TransposeKernelGenerator() = default;
  virtual std::shared_ptr<TransposeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<TransposeKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, std::vector<int64_t> &&perms);

protected:
  TransposeKernelGenerator() = default;
  TransposeKernelGenerator(const TransposeKernelGenerator &) = delete;
  TransposeKernelGenerator(TransposeKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
