#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_H_

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace cpu_transformers {
namespace kernel {

class Kernel {
public:
  Kernel() = default;
  Kernel(const Kernel &kernel) = delete;
  Kernel(Kernel &&kernel) = default;
  virtual ~Kernel() = default;

protected:
  static llvm::SmallVector<mlir::AffineMap>
  getBroadcastAffineMaps(mlir::Builder &builder,
                         llvm::ArrayRef<mlir::MemRefType> input_types,
                         const mlir::MemRefType &output_type);
  static llvm::SmallVector<mlir::AffineMap> getBroadcastMatMulAffineMaps(
      mlir::MLIRContext *context, const mlir::MemRefType &lhs_type,
      const mlir::MemRefType &rhs_type, const mlir::MemRefType &output_type);
};

class SingleInputWithoutBufferKernel : virtual public Kernel {
public:
  SingleInputWithoutBufferKernel() = default;
  SingleInputWithoutBufferKernel(const SingleInputWithoutBufferKernel &kernel) =
      delete;
  SingleInputWithoutBufferKernel(SingleInputWithoutBufferKernel &&kernel) =
      default;
  virtual ~SingleInputWithoutBufferKernel() = default;
  virtual void Run(mlir::OpBuilder &builder, mlir::Value &input,
                   mlir::Value &output) const = 0;
};

class SingleInputWithBufferKernel : virtual public Kernel {
public:
  SingleInputWithBufferKernel() = default;
  SingleInputWithBufferKernel(const SingleInputWithBufferKernel &kernel) =
      delete;
  SingleInputWithBufferKernel(SingleInputWithBufferKernel &&kernel) = default;
  virtual ~SingleInputWithBufferKernel() = default;
  virtual void Run(mlir::OpBuilder &builder, mlir::Value &input,
                   mlir::Value &output, mlir::Value &buffer) const = 0;
};

class DoubleInputsWithoutBufferKernel : virtual public Kernel {
public:
  DoubleInputsWithoutBufferKernel() = default;
  DoubleInputsWithoutBufferKernel(
      const DoubleInputsWithoutBufferKernel &kernel) = delete;
  DoubleInputsWithoutBufferKernel(DoubleInputsWithoutBufferKernel &&kernel) =
      default;
  virtual ~DoubleInputsWithoutBufferKernel() = default;
  virtual void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
                   mlir::Value &output) const = 0;
};

class DoubleInputsWithBufferKernel : virtual public Kernel {
public:
  DoubleInputsWithBufferKernel() = default;
  DoubleInputsWithBufferKernel(const DoubleInputsWithBufferKernel &kernel) =
      delete;
  DoubleInputsWithBufferKernel(DoubleInputsWithBufferKernel &&kernel) = default;
  virtual ~DoubleInputsWithBufferKernel() = default;
  virtual void Run(mlir::OpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
                   mlir::Value &output, mlir::Value &buffer) const = 0;
};

} // namespace kernel
} // namespace cpu_transformers

#endif
