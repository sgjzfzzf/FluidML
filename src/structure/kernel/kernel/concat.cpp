#include "structure/kernel/kernel/concat.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"

namespace cpu_transformers {
namespace kernel {

ConcatKernel::ConcatKernel(int64_t axis) : axis_(axis) {}

std::string Concat2Kernel::GetKernelName() const { return kKernelName; }

Concat2Kernel::Concat2Kernel(int64_t axis) : ConcatKernel(axis) {}

void Concat2Kernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                        mlir::Value &rhs, mlir::Value &output) const {
  mlir::MemRefType lhs_type = mlir::cast<mlir::MemRefType>(lhs.getType()),
                   rhs_type = mlir::cast<mlir::MemRefType>(rhs.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const int64_t rank = lhs_type.getRank();
  llvm::ArrayRef<int64_t> lhs_shape = lhs_type.getShape(),
                          rhs_shape = rhs_type.getShape(),
                          output_shape = output_type.getShape();
#ifdef DEBUG
  assert(axis_ >= 0 && axis_ < rank);
  assert(rank == rhs_type.getRank());
  assert(rank == output_type.getRank());
  const int64_t lhs_axis_dim = lhs_shape[axis_],
                rhs_axis_dim = rhs_shape[axis_],
                output_axis_dim = output_shape[axis_];
  assert(lhs_axis_dim + rhs_axis_dim == output_axis_dim);
#endif
  llvm::SmallVector<int64_t> offsets(rank, 0), strides(rank, 1);
  mlir::Value output_subview0 = builder.create<mlir::memref::SubViewOp>(
      builder.getUnknownLoc(), output, offsets, lhs_shape, strides);
  offsets[axis_] = lhs_shape[axis_];
  mlir::Value output_subview1 = builder.create<mlir::memref::SubViewOp>(
      builder.getUnknownLoc(), output, offsets, rhs_shape, strides);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{lhs},
      mlir::ValueRange{output_subview0},
      llvm::SmallVector(2, builder.getMultiDimIdentityMap(rank)),
      llvm::SmallVector<mlir::utils::IteratorType>(
          rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0], output = inputs[1];
        b.create<mlir::linalg::YieldOp>(loc, input);
      });
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{rhs},
      mlir::ValueRange{output_subview1},
      llvm::SmallVector(2, builder.getMultiDimIdentityMap(rank)),
      llvm::SmallVector<mlir::utils::IteratorType>(
          rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0], output = inputs[1];
        b.create<mlir::linalg::YieldOp>(loc, input);
      });
}

} // namespace kernel
} // namespace cpu_transformers
