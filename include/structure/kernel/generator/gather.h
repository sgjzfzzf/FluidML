#ifndef FLUIDML_STRUCTURE_KERNEL_GENERATOR_GATHER_H_
#define FLUIDML_STRUCTURE_KERNEL_GENERATOR_GATHER_H_

#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/gather.h"

namespace fluidml {
namespace kernel {

class GatherConstantIndexScalarKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~GatherConstantIndexScalarKernelGenerator() override = default;
  virtual std::shared_ptr<GatherConstantIndexScalarKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<GatherConstantIndexScalarKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, int64_t axis, int64_t index);

protected:
  GatherConstantIndexScalarKernelGenerator() = default;
  GatherConstantIndexScalarKernelGenerator(
      const GatherConstantIndexScalarKernelGenerator &) = delete;
  GatherConstantIndexScalarKernelGenerator(
      GatherConstantIndexScalarKernelGenerator &&) = default;
};

class GatherConstantIndicesTensorKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~GatherConstantIndicesTensorKernelGenerator() override = default;
  virtual std::shared_ptr<GatherConstantIndicesTensorKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<GatherConstantIndicesTensorKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, Tensor &&indices, int64_t axis);

protected:
  GatherConstantIndicesTensorKernelGenerator() = default;
  GatherConstantIndicesTensorKernelGenerator(
      const GatherConstantIndicesTensorKernelGenerator &) = delete;
  GatherConstantIndicesTensorKernelGenerator(
      GatherConstantIndicesTensorKernelGenerator &&) = default;
};

class GatherConstantDataTensorKernelGenerator
    : public SingleInputWithoutBufferKernelGenerator {
public:
  virtual ~GatherConstantDataTensorKernelGenerator() override = default;
  virtual std::shared_ptr<GatherConstantDataTensorKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<GatherConstantDataTensorKernelGenerator>
  Make(Meta &&input_meta, Meta &&output_meta, Tensor &&data);

protected:
  GatherConstantDataTensorKernelGenerator() = default;
  GatherConstantDataTensorKernelGenerator(
      const GatherConstantDataTensorKernelGenerator &) = delete;
  GatherConstantDataTensorKernelGenerator(
      GatherConstantDataTensorKernelGenerator &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
