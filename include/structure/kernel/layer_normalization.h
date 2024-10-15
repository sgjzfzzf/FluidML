#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_LAYER_NORMALIZATION_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_LAYER_NORMALIZATION_H_

#include "structure/kernel/kernel.h"
#include "structure/tensor/tensor.h"
#include "utils/float.h"

namespace cpu_transformers {
namespace kernel {

class LayerNormalizationConstantScaleBiasKernel
    : public SingleInputWithBufferKernel {
public:
  LayerNormalizationConstantScaleBiasKernel(int64_t axis, float64_t epsilon,
                                            Tensor &&scale, Tensor &&bias);
  LayerNormalizationConstantScaleBiasKernel(
      const LayerNormalizationConstantScaleBiasKernel &other) = delete;
  LayerNormalizationConstantScaleBiasKernel(
      LayerNormalizationConstantScaleBiasKernel &&other) = default;
  virtual ~LayerNormalizationConstantScaleBiasKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input, mlir::Value &output,
           mlir::Value &buffer) const override;

private:
  static constexpr char kKernelName[] =
      "LayerNormalizationConstantScaleBiasKernel";
  const int64_t axis_;
  const float64_t epsilon_;
  const Tensor scale_;
  const Tensor bias_;
};

class LayerNormalizationConstantScaleBiasKernelGenerator
    : public SingleInputWithBufferKernelGenerator {
public:
  virtual ~LayerNormalizationConstantScaleBiasKernelGenerator() = default;
  virtual std::shared_ptr<LayerNormalizationConstantScaleBiasKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) = 0;
  static std::unique_ptr<LayerNormalizationConstantScaleBiasKernelGenerator>
  Make(int64_t axis, float64_t epsilon, Tensor &&scale, Tensor &&bias);

protected:
  LayerNormalizationConstantScaleBiasKernelGenerator() = default;
  LayerNormalizationConstantScaleBiasKernelGenerator(
      const LayerNormalizationConstantScaleBiasKernelGenerator &generator) =
      delete;
  LayerNormalizationConstantScaleBiasKernelGenerator(
      LayerNormalizationConstantScaleBiasKernelGenerator &&generator) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
