#include "structure/kernel/softmax.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include <limits>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

SoftmaxKernel::SoftmaxKernel(int64_t axis) : axis_(axis) {}

void SoftmaxKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                        mlir::Value &output, mlir::Value &buffer) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_memref_type =
      mlir::cast<mlir::MemRefType>(input.getType());
  mlir::MemRefType output_memref_type =
      mlir::cast<mlir::MemRefType>(output.getType());
  mlir::MemRefType buffer_memref_type =
      mlir::cast<mlir::MemRefType>(buffer.getType());
  llvm::ArrayRef<int64_t> input_shape = input_memref_type.getShape();
  llvm::ArrayRef<int64_t> output_shape = output_memref_type.getShape();
  llvm::ArrayRef<int64_t> buffer_shape = buffer_memref_type.getShape();
  int64_t rank = input_memref_type.getRank();
#ifdef DEBUG
  assert(rank == output_memref_type.getRank());
  assert(rank > axis_);
#endif
  llvm::SmallVector<int64_t> sum_shape(rank);
  llvm::SmallVector<mlir::AffineExpr> sum_exprs;
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel),
      buffer_iterator_types;
  for (size_t i = 0; i < rank; ++i) {
    if (i == axis_) {
      sum_shape[i] = 1;
      sum_exprs.push_back(mlir::getAffineConstantExpr(0, context));
      buffer_iterator_types.push_back(mlir::utils::IteratorType::reduction);
    } else {
      sum_shape[i] = input_shape[i];
      sum_exprs.push_back(mlir::getAffineDimExpr(i, context));
      buffer_iterator_types.push_back(mlir::utils::IteratorType::parallel);
    }
  }
  mlir::AffineMap map = builder.getMultiDimIdentityMap(rank),
                  buffer_map =
                      mlir::AffineMap::get(rank, 0, sum_exprs, context);
  mlir::MemRefType sum_memref_type =
      mlir::MemRefType::get(sum_shape, input_memref_type.getElementType());
  mlir::arith::ConstantOp c0 = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));
  mlir::Value buf = builder.create<mlir::memref::ViewOp>(
      builder.getUnknownLoc(), sum_memref_type, buffer, c0, mlir::ValueRange{});
  mlir::Value c0f = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getFloatAttr(input_memref_type.getElementType(), 0));
  mlir::Value minf32 = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getFloatAttr(input_memref_type.getElementType(),
                           std::numeric_limits<float>::lowest()));
  builder.create<mlir::linalg::FillOp>(builder.getUnknownLoc(), minf32, buf);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input, buf},
      mlir::ValueRange{buf},
      llvm::SmallVector<mlir::AffineMap>{map, buffer_map, buffer_map},
      buffer_iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value input = inputs[0], value = inputs[1];
        mlir::Value max_op =
            b.create<mlir::arith::MaxNumFOp>(loc, input, value);
        b.create<mlir::linalg::YieldOp>(loc, max_op);
      });
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input, buf},
      mlir::ValueRange{output},
      llvm::SmallVector<mlir::AffineMap>{map, buffer_map, map}, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value input = inputs[0];
        mlir::Value max = inputs[1];
        mlir::Value sub_op =
            builder.create<mlir::arith::SubFOp>(loc, input, max);
        b.create<mlir::linalg::YieldOp>(loc, sub_op);
      });
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{output},
      mlir::ValueRange{output}, llvm::SmallVector<mlir::AffineMap>(2, map),
      iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0];
        mlir::Value exp_op = b.create<mlir::math::ExpOp>(loc, input);
        b.create<mlir::linalg::YieldOp>(loc, exp_op);
      });
  builder.create<mlir::linalg::FillOp>(builder.getUnknownLoc(), c0f, buf);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{output},
      mlir::ValueRange{buf}, llvm::ArrayRef<mlir::AffineMap>{map, buffer_map},
      buffer_iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0];
        mlir::Value output = inputs[1];
        mlir::Value add_op =
            builder.create<mlir::arith::AddFOp>(loc, input, output);
        b.create<mlir::linalg::YieldOp>(loc, add_op);
      });
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{output, buf},
      mlir::ValueRange{output},
      llvm::ArrayRef<mlir::AffineMap>{map, buffer_map, map}, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value input = inputs[0];
        mlir::Value sum = inputs[1];
        mlir::Value div_op =
            builder.create<mlir::arith::DivFOp>(loc, input, sum);
        b.create<mlir::linalg::YieldOp>(loc, div_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers
