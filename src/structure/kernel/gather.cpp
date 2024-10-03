#include "structure/kernel/gather.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "utils/float.h"
#include <cstddef>
#include <cstdint>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {
GatherConstantIndexScalarKernel::GatherConstantIndexScalarKernel(int64_t axis,
                                                                 int64_t index)
    : axis_(axis), index_(index) {}

void GatherConstantIndexScalarKernel::Run(mlir::OpBuilder &builder,
                                          mlir::Value &input,
                                          mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType data_memref_type =
      mlir::cast<mlir::MemRefType>(input.getType());
  mlir::MemRefType output_memref_type =
      mlir::cast<mlir::MemRefType>(output.getType());
  int64_t data_rank = data_memref_type.getRank();
  int64_t output_rank = output_memref_type.getRank();
#ifdef DEBUG
  assert(data_rank - 1 == output_rank);
#endif
  size_t axis = axis_ >= 0 ? axis_ : data_rank + axis_;
  llvm::SmallVector<mlir::AffineExpr> data_exprs;
  for (size_t i = 0; i < data_rank; ++i) {
    if (i == axis_) {
#ifdef DEBUG
      assert(index_ >= 0);
#endif
      data_exprs.push_back(mlir::getAffineConstantExpr(index_, context));
    } else {
      size_t index = i < axis_ ? i : i - 1;
      data_exprs.push_back(mlir::getAffineDimExpr(index, context));
    }
  }
  mlir::AffineMap data_map =
      mlir::AffineMap::get(output_rank, 0, data_exprs, context);
  mlir::AffineMap output_map = builder.getMultiDimIdentityMap(output_rank);
  llvm::SmallVector<mlir::AffineMap> maps = {data_map, output_map};
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types;
  for (size_t i = 0; i < output_rank; ++i) {
    iterator_types.push_back(mlir::utils::IteratorType::parallel);
  }
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value lhs = inputs[0];
        b.create<mlir::linalg::YieldOp>(loc, lhs);
      });
}

GatherConstantDataTensorKernel::GatherConstantDataTensorKernel(Tensor &&data)
    : data_(std::move(data)) {}

void GatherConstantDataTensorKernel::Run(mlir::OpBuilder &builder,
                                         mlir::Value &input,
                                         mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType indices_memref_type =
      mlir::cast<mlir::MemRefType>(input.getType());
  mlir::MemRefType output_memref_type =
      mlir::cast<mlir::MemRefType>(output.getType());
  const std::vector<int64_t> &data_shape = data_.GetShape();
  const std::vector<float64_t> &data_ref = data_.Get();
  int64_t data_rank = data_shape.size();
  llvm::ArrayRef<int64_t> indices_shape = indices_memref_type.getShape();
  int64_t indices_rank = indices_memref_type.getRank();
  llvm::ArrayRef<int64_t> output_shape = output_memref_type.getShape();
  int64_t output_rank = output_memref_type.getRank();
#ifdef DEBUG
  assert(data_rank + indices_rank - 1 == output_rank);
  for (size_t i = 0; i < output_rank; ++i) {
    if (i < indices_rank) {
      assert(indices_shape[i] == output_shape[i]);
    } else {
      assert(data_shape[i - indices_rank + 1] == output_shape[i]);
    }
  }
#endif
  mlir::ShapedType data_shaped_type =
      mlir::RankedTensorType::get(data_shape, mlir::FloatType::getF32(context));
  mlir::DenseElementsAttr elements;
  if (data_.GetType() == Type::kFloat32) {
    llvm::SmallVector<float32_t> data(data_ref.begin(), data_ref.end());
    elements =
        mlir::DenseElementsAttr::get(data_shaped_type, llvm::ArrayRef(data));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  mlir::Value data_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), elements);
  mlir::MemRefType data_memref_type = mlir::MemRefType::get(
      data_shape, mlir::FloatType::getF32(context), {}, 0);
  mlir::Value data_memref_ref = builder.create<mlir::bufferization::ToMemrefOp>(
      builder.getUnknownLoc(), data_memref_type, data_value);
  llvm::SmallVector<mlir::AffineExpr> input_exprs;
  for (size_t i = 0; i < indices_rank; ++i) {
    input_exprs.push_back(mlir::getAffineDimExpr(i, context));
  }
  mlir::AffineMap indices_map =
      mlir::AffineMap::get(output_rank, 0, input_exprs, context);
  mlir::AffineMap output_map = builder.getMultiDimIdentityMap(output_rank);
  llvm::SmallVector<mlir::AffineMap> maps = {indices_map, output_map};
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types;
  for (size_t i = 0; i < output_rank; ++i) {
    iterator_types.push_back(mlir::utils::IteratorType::parallel);
  }
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value first_index_value = b.create<mlir::arith::IndexCastOp>(
            loc, b.getIndexType(), inputs[0]);
        llvm::SmallVector<mlir::Value> indices_values;
        indices_values.push_back(first_index_value);
        for (size_t i = indices_rank; i < output_rank; ++i) {
          mlir::Value value_index =
              b.create<mlir::linalg::IndexOp>(loc, b.getIndexType(), i);
          indices_values.push_back(std::move(value_index));
        }
        mlir::Value out = b.create<mlir::memref::LoadOp>(loc, data_memref_ref,
                                                         indices_values);
        b.create<mlir::linalg::YieldOp>(loc, out);
      });
}

} // namespace kernel
} // namespace cpu_transformers
