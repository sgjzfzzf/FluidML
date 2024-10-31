#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_CONV_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_CONV_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/conv.h"

namespace fluidml {
namespace kernel {

class ConvWithoutPaddingKernelGenerator
    : public DoubleInputsWithoutBufferKernelGenerator {
public:
  virtual ~ConvWithoutPaddingKernelGenerator() = default;
  virtual std::shared_ptr<ConvWithoutPaddingKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> weights_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<ConvWithoutPaddingKernelGenerator>
  Make(Meta &&lhs_meta, Meta &&rhs_meta, Meta &&output_meta,
       std::vector<int64_t> &&dilations, int64_t group,
       std::vector<int64_t> &&kernel_shape, std::vector<int64_t> &&strides,
       std::optional<Tensor> &&bias);

protected:
  ConvWithoutPaddingKernelGenerator() = default;
  ConvWithoutPaddingKernelGenerator(const ConvWithoutPaddingKernelGenerator &) =
      delete;
  ConvWithoutPaddingKernelGenerator(ConvWithoutPaddingKernelGenerator &&) =
      default;
};

class ConvWithPaddingKernelGenerator
    : public DoubleInputsWithBufferKernelGenerator {
public:
  virtual ~ConvWithPaddingKernelGenerator() = default;
  virtual std::shared_ptr<ConvWithPaddingKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> weights_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<ConvWithPaddingKernelGenerator>
  Make(Meta &&lhs_meta, Meta &&rhs_meta, Meta &&output_meta,
       std::vector<int64_t> &&dilations, int64_t group,
       std::vector<int64_t> &&kernel_shape, std::vector<int64_t> &&pads,
       std::vector<int64_t> &&strides, std::optional<Tensor> &&bias);

protected:
  ConvWithPaddingKernelGenerator() = default;
  ConvWithPaddingKernelGenerator(const ConvWithPaddingKernelGenerator &) =
      delete;
  ConvWithPaddingKernelGenerator(ConvWithPaddingKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
