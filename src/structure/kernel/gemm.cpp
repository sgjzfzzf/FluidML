#include "structure/kernel/gemm.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "utils/float.h"
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

GemmConstantWeightsBiasKernel::GemmConstantWeightsBiasKernel(
    float64_t alpha, float64_t beta, bool transA, bool transB, Tensor &&weights,
    Tensor &&bias)
    : alpha_(alpha), beta_(beta), transA_(transA), transB_(transB),
      weights_(weights), bias_(bias) {}

void GemmConstantWeightsBiasKernel::Run(mlir::OpBuilder &builder,
                                        mlir::Value &input,
                                        mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_memref_type =
      mlir::cast<mlir::MemRefType>(input.getType());
  const int64_t input_rank = input_memref_type.getRank();
  llvm::ArrayRef<int64_t> input_shape = input_memref_type.getShape();
  Type weights_raw_type = weights_.GetType();
  const std::vector<int64_t> &weights_shape = weights_.GetShape();
  const std::vector<float64_t> &weights_ref = weights_.Get();
  const int64_t weights_rank = weights_shape.size();
  Type bias_raw_type = bias_.GetType();
  const std::vector<int64_t> &bias_shape = bias_.GetShape();
  const std::vector<float64_t> &bias_ref = bias_.Get();
  const int64_t bias_rank = bias_shape.size();
  mlir::MemRefType output_memref_type =
      mlir::cast<mlir::MemRefType>(output.getType());
  const int64_t output_rank = output_memref_type.getRank();
  llvm::ArrayRef<int64_t> output_shape = output_memref_type.getShape();
  const size_t m =
      transA_ ? input_shape[input_rank - 1] : input_shape[input_rank - 2];
  const size_t n = transB_ ? weights_shape[weights_rank - 2]
                           : weights_shape[weights_rank - 1];
  const size_t k =
      transA_ ? input_shape[input_rank - 2] : input_shape[input_rank - 1];
#ifdef DEBUG
  assert(input_rank == 2);
  assert(weights_rank == 2);
  assert(bias_rank == 1 || bias_rank == 2);
  assert(output_rank == 2);
#endif
  mlir::Value c0f = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getFloatAttr(output_memref_type.getElementType(), 0.0));
  mlir::Value alpha = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getFloatAttr(output_memref_type.getElementType(), alpha_));
  mlir::Value beta = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getFloatAttr(output_memref_type.getElementType(), beta_));
  mlir::Type weights_type = GetMLIRType(weights_raw_type, builder);
  mlir::Type bias_type = GetMLIRType(bias_raw_type, builder);
  mlir::RankedTensorType weights_shaped_type =
      mlir::RankedTensorType::get(weights_shape, weights_type);
  mlir::RankedTensorType bias_shaped_type =
      mlir::RankedTensorType::get(bias_shape, bias_type);
  mlir::DenseElementsAttr weights_elements, bias_elements;
  if (weights_raw_type == Type::kFloat32) {
    llvm::SmallVector<float32_t> weights_data(weights_ref.begin(),
                                              weights_ref.end());
    weights_elements = mlir::DenseElementsAttr::get(
        weights_shaped_type, llvm::ArrayRef(weights_data));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  if (bias_raw_type == Type::kFloat32) {
    llvm::SmallVector<float32_t> bias_data(bias_ref.begin(), bias_ref.end());
    bias_elements = mlir::DenseElementsAttr::get(bias_shaped_type,
                                                 llvm::ArrayRef(bias_data));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  mlir::Value weights_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), weights_elements);
  mlir::Value bias_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), bias_elements);
  mlir::MemRefType weights_memref_type =
      mlir::MemRefType::get(weights_shape, weights_type, {}, 0);
  mlir::MemRefType bias_memref_type =
      mlir::MemRefType::get(bias_shape, bias_type, {}, 0);
  mlir::Value weights_memref = builder.create<mlir::bufferization::ToMemrefOp>(
      builder.getUnknownLoc(), weights_memref_type, weights_value);
  mlir::Value bias_memref = builder.create<mlir::bufferization::ToMemrefOp>(
      builder.getUnknownLoc(), bias_memref_type, bias_value);
  // the order of axes is m, k, n
  llvm::SmallVector<mlir::AffineExpr>
      input_affine_exprs =
          transA_ ? llvm::SmallVector<mlir::AffineExpr>{mlir::getAffineDimExpr(
                                                            1, context),
                                                        mlir::getAffineDimExpr(
                                                            0, context)}
                  : llvm::SmallVector<mlir::AffineExpr>{mlir::getAffineDimExpr(
                                                            0, context),
                                                        mlir::getAffineDimExpr(
                                                            1, context)},
      weight_affine_exprs =
          transB_ ? llvm::SmallVector<mlir::AffineExpr>{mlir::getAffineDimExpr(
                                                            2, context),
                                                        mlir::getAffineDimExpr(
                                                            1, context)}
                  : llvm::SmallVector<mlir::AffineExpr>{mlir::getAffineDimExpr(
                                                            1, context),
                                                        mlir::getAffineDimExpr(
                                                            2, context)},
      output0_affine_exprs = {mlir::getAffineDimExpr(0, context),
                              mlir::getAffineDimExpr(2, context)},
      bias_affine_exprs,
      output1_affine_exprs = {mlir::getAffineDimExpr(0, context),
                              mlir::getAffineDimExpr(1, context)};
  if (bias_rank == 1) {
    bias_affine_exprs = {mlir::getAffineDimExpr(1, context)};
  } else if (bias_rank == 2) {
    bias_affine_exprs = {mlir::getAffineDimExpr(0, context),
                         mlir::getAffineDimExpr(1, context)};
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  mlir::AffineMap input_affine_map =
                      mlir::AffineMap::get(3, 0, input_affine_exprs, context),
                  weight_affine_map =
                      mlir::AffineMap::get(3, 0, weight_affine_exprs, context),
                  output0_affine_map =
                      mlir::AffineMap::get(3, 0, output0_affine_exprs, context),
                  bias_affine_map =
                      mlir::AffineMap::get(2, 0, bias_affine_exprs, context),
                  output1_affine_map =
                      mlir::AffineMap::get(2, 0, output1_affine_exprs, context);
  llvm::SmallVector<mlir::AffineMap> weight_affine_maps = {input_affine_map,
                                                           weight_affine_map,
                                                           output0_affine_map},
                                     bias_affine_maps = {output1_affine_map,
                                                         bias_affine_map,
                                                         output1_affine_map};
  llvm::SmallVector<mlir::utils::IteratorType>
      weight_iterator_types = {mlir::utils::IteratorType::parallel,
                               mlir::utils::IteratorType::reduction,
                               mlir::utils::IteratorType::parallel},
      bias_iterator_types = {mlir::utils::IteratorType::parallel,
                             mlir::utils::IteratorType::parallel};
  builder.create<mlir::linalg::FillOp>(builder.getUnknownLoc(), c0f, output);
  // TODO: potential bugs, the types of weights and output are not float.
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{input, weights_memref}, mlir::ValueRange{output},
      weight_affine_maps, weight_iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value input = inputs[0], weights = inputs[1], output = inputs[2];
        mlir::Value mul_op = b.create<mlir::arith::MulFOp>(loc, input, weights);
        mlir::Value add_op = b.create<mlir::arith::AddFOp>(loc, mul_op, output);
        b.create<mlir::linalg::YieldOp>(loc, add_op);
      });
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{output, bias_memref}, mlir::ValueRange{output},
      bias_affine_maps, bias_iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value output = inputs[0], weights = inputs[1];
        mlir::Value mul0_op =
            b.create<mlir::arith::MulFOp>(loc, alpha, weights);
        mlir::Value mul1_op = b.create<mlir::arith::MulFOp>(loc, beta, output);
        mlir::Value add_op =
            b.create<mlir::arith::AddFOp>(loc, mul0_op, mul1_op);
        b.create<mlir::linalg::YieldOp>(loc, add_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers
