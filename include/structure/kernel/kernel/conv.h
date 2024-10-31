#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_CONV_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_CONV_H_

#include "structure/kernel/kernel/kernel.h"
#include "structure/tensor/tensor.h"

namespace fluidml {
namespace kernel {

class ConvKernel : virtual public Kernel {
public:
  ConvKernel(std::vector<int64_t> &&dilations, int64_t group,
             std::vector<int64_t> &&kernel_shape,
             std::vector<int64_t> &&strides, std::optional<Tensor> &&bias);
  ConvKernel(const ConvKernel &) = delete;
  ConvKernel(ConvKernel &&) = default;
  virtual ~ConvKernel() = default;

protected:
  void run(mlir::OpBuilder &builder, mlir::Value &input, mlir::Value &weights,
           mlir::Value &output) const;
  const std::vector<int64_t> dilations_;
  const int64_t group_;
  const std::vector<int64_t> kernel_shape_;
  const std::vector<int64_t> strides_;
  const std::optional<Tensor> bias_;
};

class ConvWithoutPaddingKernel : public ConvKernel,
                                 public DoubleInputsWithoutBufferKernel {
public:
  static constexpr char kKernelName[] = "ConvWithoutPaddingKernel";
  ConvWithoutPaddingKernel(std::vector<int64_t> &&dilations, int64_t group,
                           std::vector<int64_t> &&kernel_shape,
                           std::vector<int64_t> &&strides,
                           std::optional<Tensor> &&bias);
  ConvWithoutPaddingKernel(const ConvWithoutPaddingKernel &) = delete;
  ConvWithoutPaddingKernel(ConvWithoutPaddingKernel &&) = default;
  virtual ~ConvWithoutPaddingKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output) const override;
};

class ConvWithPaddingKernel : public ConvKernel,
                              public DoubleInputsWithBufferKernel {
public:
  static constexpr char kKernelName[] = "ConvWithPaddingKernel";
  ConvWithPaddingKernel(std::vector<int64_t> &&dilations, int64_t group,
                        std::vector<int64_t> &&kernel_shape,
                        std::vector<int64_t> &&pads,
                        std::vector<int64_t> &&strides,
                        std::optional<Tensor> &&bias);
  ConvWithPaddingKernel(const ConvWithPaddingKernel &) = delete;
  ConvWithPaddingKernel(ConvWithPaddingKernel &&) = default;
  virtual ~ConvWithPaddingKernel() = default;
  std::string GetKernelName() const override;
  void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
           mlir::Value &output, mlir::Value &buffer) const override;

private:
  const std::vector<int64_t> pads_;
};

} // namespace kernel
} // namespace fluidml

#endif
