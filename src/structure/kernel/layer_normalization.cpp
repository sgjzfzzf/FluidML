#include "structure/kernel/layer_normalization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "utils/float.h"
#include "utils/type.h"
#include "llvm/ADT/ArrayRef.h"
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

LayerNormalizationConstantScaleBiasKernel::
    LayerNormalizationConstantScaleBiasKernel(int64_t axis, float64_t epsilon,
                                              Tensor &&scale, Tensor &&bias)
    : axis_(axis), epsilon_(epsilon), scale_(std::move(scale)),
      bias_(std::move(bias)) {}

void LayerNormalizationConstantScaleBiasKernel::Run(mlir::OpBuilder &builder,
                                                    mlir::Value &input,
                                                    mlir::Value &output,
                                                    mlir::Value &buffer) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType());
  const Meta &scale_meta = scale_.GetMeta();
  const Meta &bias_meta = bias_.GetMeta();
  const std::vector<float64_t> &scale_data = scale_.Get();
  const std::vector<float64_t> &bias_data = bias_.Get();
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const int64_t rank = input_type.getRank();
  llvm::ArrayRef<int64_t> shape = input_type.getShape(),
                          scale_shape = scale_meta.GetShape(),
                          bias_shape = bias_meta.GetShape();
  const int64_t target_dim = shape[axis_], scale_size = scale_shape.size(),
                bias_size = bias_shape.size();
#ifdef DEBUG
  llvm::ArrayRef<int64_t> output_shape = output_type.getShape();
  assert(rank == shape.size());
  assert(rank == output_shape.size());
  assert(shape == output_shape);
  // Only support float32 for now.
  assert(scale_meta.GetType() == Type::FLOAT32);
  assert(bias_meta.GetType() == Type::FLOAT32);
#endif
  llvm::SmallVector<int64_t> buffer_shape;
  llvm::SmallVector<mlir::AffineExpr> buffer_exprs, scale_exprs, bias_exprs;
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel),
      buffer_iterator_types;
  for (size_t i = 0; i < rank; ++i) {
    const int64_t scale_index = scale_size - rank + i,
                  bias_index = bias_size - rank + i;
    if (i == axis_) {
      buffer_shape.push_back(1);
      buffer_exprs.push_back(builder.getAffineConstantExpr(0));
      buffer_iterator_types.push_back(mlir::utils::IteratorType::reduction);
    } else {
      buffer_shape.push_back(shape[i]);
      buffer_exprs.push_back(builder.getAffineDimExpr(i));
      buffer_iterator_types.push_back(mlir::utils::IteratorType::parallel);
    }
    if (scale_index >= 0) {
      scale_exprs.push_back(builder.getAffineDimExpr(i));
    }
    if (bias_index >= 0) {
      bias_exprs.push_back(builder.getAffineDimExpr(i));
    }
  }
  mlir::AffineMap map = builder.getMultiDimIdentityMap(rank),
                  buffer_map =
                      mlir::AffineMap::get(rank, 0, buffer_exprs, context),
                  scale_map =
                      mlir::AffineMap::get(rank, 0, scale_exprs, context),
                  bias_map = mlir::AffineMap::get(rank, 0, bias_exprs, context);
  mlir::MemRefType buffer_type = mlir::MemRefType::get(
                       buffer_shape, input_type.getElementType()),
                   scale_type = mlir::MemRefType::get(
                       scale_shape, input_type.getElementType()),
                   bias_type = mlir::MemRefType::get(
                       bias_shape, input_type.getElementType());
  llvm::SmallVector<float32_t> scale_data_f32(scale_data.begin(),
                                              scale_data.end()),
      bias_data_f32(bias_data.begin(), bias_data.end());
  mlir::RankedTensorType scale_tensor_type = mlir::RankedTensorType::get(
                             scale_shape, input_type.getElementType()),
                         bias_tensor_type = mlir::RankedTensorType::get(
                             bias_shape, input_type.getElementType());
  mlir::DenseElementsAttr scale_attr = mlir::DenseElementsAttr::get(
                              scale_tensor_type,
                              llvm::ArrayRef<float32_t>(scale_data_f32)),
                          bias_attr = mlir::DenseElementsAttr::get(
                              bias_tensor_type,
                              llvm::ArrayRef<float32_t>(bias_data_f32));
  mlir::Value scale_constant = builder.create<mlir::arith::ConstantOp>(
                  builder.getUnknownLoc(), scale_attr),
              bias_constant = builder.create<mlir::arith::ConstantOp>(
                  builder.getUnknownLoc(), bias_attr);
  mlir::Value c0index = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getIndexType(), builder.getIndexAttr(0));
  mlir::Value c0f = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getFloatAttr(input_type.getElementType(), 0));
  mlir::Value c1f = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getFloatAttr(input_type.getElementType(), 1));
  mlir::Value buffer_view = builder.create<mlir::memref::ViewOp>(
                  builder.getUnknownLoc(), buffer_type, buffer, c0index,
                  mlir::ValueRange{}),
              scale_view = builder.create<mlir::bufferization::ToMemrefOp>(
                  builder.getUnknownLoc(), scale_type, scale_constant),
              bias_view = builder.create<mlir::bufferization::ToMemrefOp>(
                  builder.getUnknownLoc(), bias_type, bias_constant);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{buffer_view},
      llvm::ArrayRef<mlir::AffineMap>{map, buffer_map}, buffer_iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0], output = inputs[1];
        mlir::Value add_op = b.create<mlir::arith::AddFOp>(loc, input, output);
        b.create<mlir::linalg::YieldOp>(loc, add_op);
      });
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{buffer_view},
      mlir::ValueRange{buffer_view}, llvm::ArrayRef<mlir::AffineMap>{map, map},
      iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0];
        mlir::Value num = b.create<mlir::arith::ConstantOp>(
            loc, input.getType(),
            builder.getFloatAttr(input.getType(), target_dim));
        mlir::Value div_op = b.create<mlir::arith::DivFOp>(loc, input, num);
        b.create<mlir::linalg::YieldOp>(loc, div_op);
      });
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{input, buffer_view}, mlir::ValueRange{output},
      llvm::ArrayRef<mlir::AffineMap>{map, buffer_map, map}, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value input = inputs[0], mean = inputs[1];
        mlir::Value sub_op = b.create<mlir::arith::SubFOp>(loc, input, mean);
        b.create<mlir::linalg::YieldOp>(loc, sub_op);
      });
  builder.create<mlir::linalg::FillOp>(builder.getUnknownLoc(), c0f,
                                       buffer_view);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{output},
      mlir::ValueRange{buffer_view},
      llvm::ArrayRef<mlir::AffineMap>{map, buffer_map}, buffer_iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0], output = inputs[1];
        mlir::Value mul_op = b.create<mlir::arith::MulFOp>(loc, input, input);
        mlir::Value add_op = b.create<mlir::arith::AddFOp>(loc, output, mul_op);
        b.create<mlir::linalg::YieldOp>(loc, add_op);
      });
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{buffer_view},
      mlir::ValueRange{buffer_view}, llvm::ArrayRef<mlir::AffineMap>{map, map},
      iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0];
        mlir::Value num = b.create<mlir::arith::ConstantOp>(
            loc, input.getType(),
            builder.getFloatAttr(input.getType(), target_dim));
        mlir::Value div_op = b.create<mlir::arith::DivFOp>(loc, input, num);
        b.create<mlir::linalg::YieldOp>(loc, div_op);
      });
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{output, buffer_view, scale_view, bias_view},
      mlir::ValueRange{output},
      llvm::ArrayRef<mlir::AffineMap>{map, buffer_map, scale_map, bias_map,
                                      map},
      iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 5);
#endif
        mlir::Value input = inputs[0], var = inputs[1], scale = inputs[2],
                    bias = inputs[3];
        mlir::Value epsilon = b.create<mlir::arith::ConstantOp>(
            loc, input.getType(),
            builder.getFloatAttr(input.getType(), epsilon_));
        mlir::Value add0_op = b.create<mlir::arith::AddFOp>(loc, var, epsilon);
        mlir::Value sqrt_op = b.create<mlir::math::SqrtOp>(loc, add0_op);
        mlir::Value div_op = b.create<mlir::arith::DivFOp>(loc, c1f, sqrt_op);
        mlir::Value mul0_op = b.create<mlir::arith::MulFOp>(loc, input, div_op);
        mlir::Value mul1_op =
            b.create<mlir::arith::MulFOp>(loc, mul0_op, scale);
        mlir::Value add1_op = b.create<mlir::arith::AddFOp>(loc, mul1_op, bias);
        b.create<mlir::linalg::YieldOp>(loc, add1_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers
