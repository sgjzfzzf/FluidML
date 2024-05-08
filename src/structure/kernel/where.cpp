#include "structure/kernel/where.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "utils/float.h"
#include "utils/type.h"
#include <cstdint>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

void WhereConstantCondConstantScalarYKernel::Run(mlir::OpBuilder &builder,
                                                 const Tensor &cond,
                                                 mlir::Value &x, Type type,
                                                 float64_t y,
                                                 mlir::Value &output) {
  mlir::MLIRContext *context = builder.getContext();
  const std::vector<int64_t> cond_shape = cond.GetShape();
  const std::vector<float64_t> cond_ref = cond.Get();
  mlir::MemRefType x_type = mlir::cast<mlir::MemRefType>(x.getType());
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = output_type.getRank();
#ifdef DEBUG
  assert(cond.GetType() == Type::BOOL);
  // TODO: add support for other types in the future
  assert(type == Type::FLOAT32);
#endif
  mlir::RankedTensorType cond_tensor_type =
      mlir::RankedTensorType::get(cond_shape, builder.getI1Type());
  mlir::MemRefType cond_memref_type =
      mlir::MemRefType::get(cond_shape, builder.getI1Type());
  llvm::SmallVector<bool> cond_data(cond_ref.begin(), cond_ref.end());
  mlir::DenseElementsAttr cond_elements =
      mlir::DenseElementsAttr::get(cond_tensor_type, cond_data);
  mlir::arith::ConstantOp cond_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), cond_elements);
  mlir::arith::ConstantOp y_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getF32FloatAttr(y));
  mlir::bufferization::ToMemrefOp cond_memref =
      builder.create<mlir::bufferization::ToMemrefOp>(
          builder.getUnknownLoc(), cond_memref_type, cond_value);
  llvm::SmallVector<mlir::AffineMap> maps = getBroadcastAffineMaps(
      builder, llvm::ArrayRef<mlir::MemRefType>{cond_memref_type, x_type},
      output_type);
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{cond_memref, x}, mlir::ValueRange{output}, maps,
      iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value select = inputs[0], x = inputs[1];
        mlir::Value select_op =
            b.create<mlir::arith::SelectOp>(loc, select, x, y_value);
        b.create<mlir::linalg::YieldOp>(loc, select_op);
      });
}

void WhereConstantCondConstantTensorYKernel::Run(mlir::OpBuilder &builder,
                                                 const Tensor &cond,
                                                 mlir::Value &x,
                                                 const Tensor &y,
                                                 mlir::Value &output) {
  mlir::MLIRContext *context = builder.getContext();
  const std::vector<int64_t> cond_shape = cond.GetShape();
  const std::vector<int64_t> y_shape = y.GetShape();
  const std::vector<float64_t> cond_ref = cond.Get();
  const std::vector<float64_t> y_ref = y.Get();
  mlir::MemRefType x_type = mlir::cast<mlir::MemRefType>(x.getType());
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = output_type.getRank();
#ifdef DEBUG
  assert(cond.GetType() == Type::BOOL);
  // TODO: add support for other types in the future
  assert(y.GetType() == Type::FLOAT32);
#endif
  llvm::SmallVector<bool> cond_data(cond_ref.begin(), cond_ref.end());
  mlir::RankedTensorType cond_tensor_type =
      mlir::RankedTensorType::get(cond_shape, builder.getI1Type());
  mlir::MemRefType cond_memref_type =
      mlir::MemRefType::get(cond_shape, builder.getI1Type());
  mlir::DenseElementsAttr cond_elements =
      mlir::DenseElementsAttr::get(cond_tensor_type, cond_data);
  mlir::arith::ConstantOp cond_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), cond_elements);
  mlir::bufferization::ToMemrefOp cond_memref =
      builder.create<mlir::bufferization::ToMemrefOp>(
          builder.getUnknownLoc(), cond_memref_type, cond_value);
  llvm::SmallVector<float32_t> y_data(y_ref.begin(), y_ref.end());
  mlir::RankedTensorType y_tensor_type =
      mlir::RankedTensorType::get(y_shape, builder.getF32Type());
  mlir::MemRefType y_memref_type =
      mlir::MemRefType::get(y_shape, builder.getF32Type());
  mlir::DenseElementsAttr y_elements = mlir::DenseElementsAttr::get(
      y_tensor_type, llvm::ArrayRef<float32_t>(y_data));
  mlir::arith::ConstantOp y_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), y_elements);
  mlir::bufferization::ToMemrefOp y_memref =
      builder.create<mlir::bufferization::ToMemrefOp>(builder.getUnknownLoc(),
                                                      y_memref_type, y_value);
  llvm::SmallVector<mlir::AffineMap> maps = getBroadcastAffineMaps(
      builder, llvm::ArrayRef<mlir::MemRefType>{cond_memref_type, x_type},
      output_type);
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{cond_memref, x, y_memref}, mlir::ValueRange{output},
      maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 4);
#endif
        mlir::Value select = inputs[0], x = inputs[1], y = inputs[2];
        mlir::Value select_op =
            b.create<mlir::arith::SelectOp>(loc, select, x, y);
        b.create<mlir::linalg::YieldOp>(loc, select_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers
