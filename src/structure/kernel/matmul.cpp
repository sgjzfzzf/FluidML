#include "structure/kernel/matmul.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "utils/float.h"
#include "utils/type.h"
#include <cstdint>
#ifdef DEBUG
#include "exception/unreachable_exception.h"
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

void MatMulKernel::run(mlir::OpBuilder &builder, mlir::Value &lhs,
                       mlir::Value &rhs, mlir::Value &output) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType lhs_type = mlir::cast<mlir::MemRefType>(lhs.getType());
  mlir::MemRefType rhs_type = mlir::cast<mlir::MemRefType>(rhs.getType());
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  llvm::SmallVector<mlir::AffineMap> maps =
      getBroadcastMatMulAffineMaps(context, lhs_type, rhs_type, output_type);
  const int64_t rank = output_type.getRank();
#ifdef DEBUG
  assert(rank >= 2);
#endif
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank + 1, mlir::utils::IteratorType::parallel);
  iterator_types[rank - 1] = mlir::utils::IteratorType::reduction;
  mlir::Value c0f = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getFloatAttr(output_type.getElementType(), 0.0));
  builder.create<mlir::linalg::FillOp>(builder.getUnknownLoc(), c0f, output);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value lhs = inputs[0], rhs = inputs[1], output = inputs[2];
        // TODO: potential bugs, the types of lhs, rhs, and output are not
        // float.
        mlir::Value mul_op = b.create<mlir::arith::MulFOp>(loc, lhs, rhs);
        mlir::Value add_op = b.create<mlir::arith::AddFOp>(loc, mul_op, output);
        b.create<mlir::linalg::YieldOp>(loc, add_op);
      });
}

void MatMulConstantLhsKernel::Run(mlir::OpBuilder &builder, const Tensor &lhs,
                                  mlir::Value &rhs, mlir::Value &output) {
  mlir::MLIRContext *context = builder.getContext();
  Type lhs_raw_type = lhs.GetType();
  mlir::Type lhs_type = GetMLIRType(lhs_raw_type, builder);
  const std::vector<int64_t> &lhs_shape = lhs.GetShape();
  const std::vector<float64_t> &lhs_ref = lhs.Get();
  mlir::RankedTensorType lhs_tensor_type =
      mlir::RankedTensorType::get(lhs_shape, lhs_type);
  mlir::DenseElementsAttr lhs_elements;
  if (lhs_raw_type == Type::FLOAT32) {
    std::vector<float32_t> lhs_data(lhs_ref.begin(), lhs_ref.end());
    lhs_elements = mlir::DenseElementsAttr::get(
        lhs_tensor_type, llvm::ArrayRef<float32_t>(lhs_data));
  } else {
#ifdef DEBUG
    throw UnreachableException();
#else
    __builtin_unreachable();
#endif
  }
  mlir::Value lhs_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), lhs_elements);
  mlir::MemRefType lhs_memref_type = mlir::MemRefType::get(lhs_shape, lhs_type);
  mlir::Value lhs_buffer = builder.create<mlir::bufferization::ToMemrefOp>(
      builder.getUnknownLoc(), lhs_memref_type, lhs_value);
  run(builder, lhs_buffer, rhs, output);
}

void MatMulConstantRhsKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                                  const Tensor &rhs, mlir::Value &output) {
  mlir::MLIRContext *context = builder.getContext();
  Type rhs_raw_type = rhs.GetType();
  mlir::Type rhs_type = GetMLIRType(rhs_raw_type, builder);
  const std::vector<int64_t> &rhs_shape = rhs.GetShape();
  const std::vector<float64_t> &rhs_ref = rhs.Get();
  mlir::RankedTensorType rhs_tensor_type =
      mlir::RankedTensorType::get(rhs_shape, rhs_type);
  mlir::DenseElementsAttr rhs_elements;
  if (rhs_raw_type == Type::FLOAT32) {
    std::vector<float32_t> rhs_data(rhs_ref.begin(), rhs_ref.end());
    rhs_elements = mlir::DenseElementsAttr::get(
        rhs_tensor_type, llvm::ArrayRef<float32_t>(rhs_data));
  } else {
#ifdef DEBUG
    throw UnreachableException();
#else
    __builtin_unreachable();
#endif
  }
  mlir::Value rhs_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), rhs_elements);
  mlir::MemRefType rhs_memref_type = mlir::MemRefType::get(rhs_shape, rhs_type);
  mlir::Value rhs_buffer = builder.create<mlir::bufferization::ToMemrefOp>(
      builder.getUnknownLoc(), rhs_memref_type, rhs_value);
  run(builder, lhs, rhs_buffer, output);
}

void MatMulCommonKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                             mlir::Value &rhs, mlir::Value &output) {
  run(builder, lhs, rhs, output);
}

} // namespace kernel
} // namespace cpu_transformers
