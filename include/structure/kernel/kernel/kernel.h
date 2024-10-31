#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_KERNEL_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_KERNEL_H_

#include "mlir/IR/Builders.h"

namespace fluidml {
namespace kernel {

class Kernel {
public:
  virtual ~Kernel() = default;
  virtual std::string GetKernelName() const = 0;

protected:
  Kernel() = default;
  Kernel(const Kernel &) = delete;
  Kernel(Kernel &&) = default;
};

class SingleInputKernel : virtual public Kernel {
public:
  virtual ~SingleInputKernel() = default;

protected:
  SingleInputKernel() = default;
  SingleInputKernel(const SingleInputKernel &) = delete;
  SingleInputKernel(SingleInputKernel &&) = default;
};

class DoubleInputsKernel : virtual public Kernel {
public:
  virtual ~DoubleInputsKernel() = default;

protected:
  DoubleInputsKernel() = default;
  DoubleInputsKernel(const DoubleInputsKernel &) = delete;
  DoubleInputsKernel(DoubleInputsKernel &&) = default;
};

class SingleInputWithoutBufferKernel : public SingleInputKernel {
public:
  virtual ~SingleInputWithoutBufferKernel() = default;
  virtual void Run(mlir::OpBuilder &builder, mlir::Value &input,
                   mlir::Value &output) const = 0;

protected:
  SingleInputWithoutBufferKernel() = default;
  SingleInputWithoutBufferKernel(const SingleInputWithoutBufferKernel &) =
      delete;
  SingleInputWithoutBufferKernel(SingleInputWithoutBufferKernel &&) = default;
};

class SingleInputWithBufferKernel : public SingleInputKernel {
public:
  virtual ~SingleInputWithBufferKernel() = default;
  virtual void Run(mlir::OpBuilder &builder, mlir::Value &input,
                   mlir::Value &output, mlir::Value &buffer) const = 0;

protected:
  SingleInputWithBufferKernel() = default;
  SingleInputWithBufferKernel(const SingleInputWithBufferKernel &) = delete;
  SingleInputWithBufferKernel(SingleInputWithBufferKernel &&) = default;
};

class DoubleInputsWithoutBufferKernel : public DoubleInputsKernel {
public:
  virtual ~DoubleInputsWithoutBufferKernel() = default;
  virtual void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
                   mlir::Value &output) const = 0;

protected:
  DoubleInputsWithoutBufferKernel() = default;
  DoubleInputsWithoutBufferKernel(const DoubleInputsWithoutBufferKernel &) =
      delete;
  DoubleInputsWithoutBufferKernel(DoubleInputsWithoutBufferKernel &&) = default;
};

class DoubleInputsWithBufferKernel : public DoubleInputsKernel {
public:
  virtual ~DoubleInputsWithBufferKernel() = default;
  virtual void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
                   mlir::Value &output, mlir::Value &buffer) const = 0;

protected:
  DoubleInputsWithBufferKernel() = default;
  DoubleInputsWithBufferKernel(const DoubleInputsWithBufferKernel &) = delete;
  DoubleInputsWithBufferKernel(DoubleInputsWithBufferKernel &&) = default;
};

} // namespace kernel
} // namespace fluidml

#endif
