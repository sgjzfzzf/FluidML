#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_MAXPOOL_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_MAXPOOL_H_

#include "structure/kernel/kernel/kernel.h"

namespace fluidml {
namespace kernel {

class MaxPoolKernel : virtual public Kernel {
public:
  MaxPoolKernel(std::vector<int64_t> &&kernel_shape,
                std::vector<int64_t> &&strides);
  MaxPoolKernel(const MaxPoolKernel &) = delete;
  MaxPoolKernel(MaxPoolKernel &&) = default;
  virtual ~MaxPoolKernel() = default;

protected:
  void run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const;
  const std::vector<int64_t> kernel_shape_;
  const std::vector<int64_t> strides_;
};

class MaxPoolWithoutPaddingKernel : public MaxPoolKernel,
                                    public SingleInputWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "MaxPoolWithoutPaddingKernel";
  MaxPoolWithoutPaddingKernel(std::vector<int64_t> &&kernel_shape,
                              std::vector<int64_t> &&strides);
  MaxPoolWithoutPaddingKernel(const MaxPoolWithoutPaddingKernel &) = delete;
  MaxPoolWithoutPaddingKernel(MaxPoolWithoutPaddingKernel &&) = default;
  virtual ~MaxPoolWithoutPaddingKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &input,
           mlir::Value &output) const override;
};

} // namespace kernel
} // namespace fluidml

#endif
