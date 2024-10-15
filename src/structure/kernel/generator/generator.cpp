#include "structure/kernel/generator/generator.h"

namespace cpu_transformers {
namespace kernel {

std::shared_ptr<SingleInputKernel>
SingleInputWithoutBufferKernelGenerator::YieldSingleInputKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return YieldSingleInputWithoutBufferKernel(input_layout, output_layout);
}

std::shared_ptr<SingleInputKernel>
SingleInputWithBufferKernelGenerator::YieldSingleInputKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return YieldSingleInputWithBufferKernel(input_layout, output_layout);
}

std::shared_ptr<DoubleInputsKernel>
DoubleInputsWithoutBufferKernelGenerator::YieldDoubleInputsKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return YieldDoubleInputsWithoutBufferKernel(lhs_layout, rhs_layout,
                                              output_layout);
}

std::shared_ptr<DoubleInputsKernel>
DoubleInputsWithBufferKernelGenerator::YieldDoubleInputsKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return YieldDoubleInputsWithBufferKernel(lhs_layout, rhs_layout,
                                           output_layout);
}

} // namespace kernel
} // namespace cpu_transformers
