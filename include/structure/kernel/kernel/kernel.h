#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_H_

#include "mlir/IR/Builders.h"

namespace cpu_transformers {
namespace kernel {

class Kernel {
public:
  virtual ~Kernel() = default;
  virtual std::string GetKernelName() const = 0;

protected:
  Kernel() = default;
  Kernel(const Kernel &kernel) = delete;
  Kernel(Kernel &&kernel) = default;
};

class SingleInputKernel : virtual public Kernel {
public:
  virtual ~SingleInputKernel() = default;

protected:
  SingleInputKernel() = default;
  SingleInputKernel(const SingleInputKernel &kernel) = delete;
  SingleInputKernel(SingleInputKernel &&kernel) = default;
};

class DoubleInputsKernel : virtual public Kernel {
public:
  virtual ~DoubleInputsKernel() = default;

protected:
  DoubleInputsKernel() = default;
  DoubleInputsKernel(const DoubleInputsKernel &kernel) = delete;
  DoubleInputsKernel(DoubleInputsKernel &&kernel) = default;
};

class SingleInputWithoutBufferKernel : public SingleInputKernel {
public:
  virtual ~SingleInputWithoutBufferKernel() = default;
  virtual void Run(mlir::OpBuilder &builder, mlir::Value &input,
                   mlir::Value &output) const = 0;

protected:
  SingleInputWithoutBufferKernel() = default;
  SingleInputWithoutBufferKernel(const SingleInputWithoutBufferKernel &kernel) =
      delete;
  SingleInputWithoutBufferKernel(SingleInputWithoutBufferKernel &&kernel) =
      default;
};

class SingleInputWithBufferKernel : public SingleInputKernel {
public:
  virtual ~SingleInputWithBufferKernel() = default;
  virtual void Run(mlir::OpBuilder &builder, mlir::Value &input,
                   mlir::Value &output, mlir::Value &buffer) const = 0;

protected:
  SingleInputWithBufferKernel() = default;
  SingleInputWithBufferKernel(const SingleInputWithBufferKernel &kernel) =
      delete;
  SingleInputWithBufferKernel(SingleInputWithBufferKernel &&kernel) = default;
};

class DoubleInputsWithoutBufferKernel : public DoubleInputsKernel {
public:
  virtual ~DoubleInputsWithoutBufferKernel() = default;
  virtual void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
                   mlir::Value &output) const = 0;

protected:
  DoubleInputsWithoutBufferKernel() = default;
  DoubleInputsWithoutBufferKernel(
      const DoubleInputsWithoutBufferKernel &kernel) = delete;
  DoubleInputsWithoutBufferKernel(DoubleInputsWithoutBufferKernel &&kernel) =
      default;
};

class DoubleInputsWithBufferKernel : public DoubleInputsKernel {
public:
  virtual ~DoubleInputsWithBufferKernel() = default;
  virtual void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
                   mlir::Value &output, mlir::Value &buffer) const = 0;

protected:
  DoubleInputsWithBufferKernel() = default;
  DoubleInputsWithBufferKernel(const DoubleInputsWithBufferKernel &kernel) =
      delete;
  DoubleInputsWithBufferKernel(DoubleInputsWithBufferKernel &&kernel) = default;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
