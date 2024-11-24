#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_AVERAGEPOOL_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_AVERAGEPOOL_H_

#include "structure/kernel/kernel/kernel.h"

namespace fluidml {
namespace kernel {

class AveragePoolKernel : virtual public SingleInputWithoutBufferKernel {
public:
  AveragePoolKernel(std::vector<int64_t> &&dilations,
                    std::vector<int64_t> &&kernel_shape,
                    std::vector<int64_t> &&strides);
  AveragePoolKernel(const AveragePoolKernel &) = delete;
  AveragePoolKernel(AveragePoolKernel &&) = default;
  virtual ~AveragePoolKernel() = default;

protected:
  void run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const;
  const std::vector<int64_t> dilations_;
  const std::vector<int64_t> kernel_shape_;
  const std::vector<int64_t> strides_;
};

class AveragePoolWithoutPaddingKernel : public AveragePoolKernel {
public:
  static constexpr char kKernelName[] = "AveragePoolWithoutPaddingKernel";
  AveragePoolWithoutPaddingKernel(std::vector<int64_t> &&dilations,
                                  std::vector<int64_t> &&kernel_shape,
                                  std::vector<int64_t> &&strides);
  AveragePoolWithoutPaddingKernel(const AveragePoolWithoutPaddingKernel &) =
      delete;
  AveragePoolWithoutPaddingKernel(AveragePoolWithoutPaddingKernel &&) = default;
  virtual ~AveragePoolWithoutPaddingKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace fluidml

#endif
