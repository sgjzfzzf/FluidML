#include "structure/kernel/kernel/gemm.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "structure/kernel/kernel/utils.h"
#include "utils/float.h"
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

GemmConstantBiasKernel::GemmConstantBiasKernel(float64_t alpha, float64_t beta,
                                               bool transA, bool transB,
                                               Tensor &&bias)
    : alpha_(alpha), beta_(beta), transA_(transA), transB_(transB), bias_(bias),
      axes_({Axis::i, Axis::j, Axis::k}) {}

GemmConstantBiasKernel::GemmConstantBiasKernel(
    float64_t alpha, float64_t beta, bool transA, bool transB, Tensor &&bias,
    llvm::SmallVector<Axis, 3> &&axes)
    : alpha_(alpha), beta_(beta), transA_(transA), transB_(transB), bias_(bias),
      axes_(std::move(axes)) {}

std::string GemmConstantBiasKernel::GetKernelName() const {
  return kKernelName;
}

llvm::ArrayRef<Axis> GemmConstantBiasKernel::GetAxes() { return axes_; }

void GemmConstantBiasKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                                 mlir::Value &rhs, mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType lhs_memref_type =
                       mlir::cast<mlir::MemRefType>(lhs.getType()),
                   rhs_memref_type =
                       mlir::cast<mlir::MemRefType>(rhs.getType());
  const int64_t lhs_rank = lhs_memref_type.getRank();
  llvm::ArrayRef<int64_t> lhs_shape = lhs_memref_type.getShape(),
                          rhs_shape = rhs_memref_type.getShape();
  const std::vector<int64_t> bias_shape = bias_.GetShape();
  const std::vector<float64_t> &bias_ref = bias_.Get();
  const int64_t rhs_rank = rhs_shape.size(), bias_rank = bias_shape.size();
  Type bias_raw_type = bias_.GetType();
  mlir::MemRefType output_memref_type =
      mlir::cast<mlir::MemRefType>(output.getType());
  const int64_t output_rank = output_memref_type.getRank();
#ifdef DEBUG
  assert(lhs_rank == 2);
  assert(rhs_rank == 2);
  assert(bias_rank == 1 || bias_rank == 2);
  assert(output_rank == 2);
#endif
  mlir::Value c0f = builder.create<mlir::arith::ConstantOp>(
                  builder.getUnknownLoc(),
                  builder.getFloatAttr(output_memref_type.getElementType(),
                                       0.0)),
              alpha = builder.create<mlir::arith::ConstantOp>(
                  builder.getUnknownLoc(),
                  builder.getFloatAttr(output_memref_type.getElementType(),
                                       alpha_)),
              beta = builder.create<mlir::arith::ConstantOp>(
                  builder.getUnknownLoc(),
                  builder.getFloatAttr(output_memref_type.getElementType(),
                                       beta_));
  mlir::Type bias_type = GetMLIRType(bias_raw_type, builder);
  mlir::RankedTensorType bias_shaped_type =
      mlir::RankedTensorType::get(bias_shape, bias_type);
  mlir::DenseElementsAttr bias_elements;
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
  mlir::Value bias_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), bias_elements);
  mlir::MemRefType bias_memref_type =
      mlir::MemRefType::get(bias_shape, bias_type, {}, 0);
  mlir::Value bias_memref = builder.create<mlir::bufferization::ToMemrefOp>(
      builder.getUnknownLoc(), bias_memref_type, bias_value);
  const Axis x0 = axes_[0], x1 = axes_[1], x2 = axes_[2];
  llvm::SmallVector<mlir::AffineExpr>
      lhs_affine_exprs =
          transA_ ? llvm::SmallVector<mlir::AffineExpr>{mlir::getAffineDimExpr(
                                                            x2, context),
                                                        mlir::getAffineDimExpr(
                                                            x0, context)}
                  : llvm::SmallVector<mlir::AffineExpr>{mlir::getAffineDimExpr(
                                                            x0, context),
                                                        mlir::getAffineDimExpr(
                                                            x2, context)},
      rhs_affine_exprs =
          transB_ ? llvm::SmallVector<mlir::AffineExpr>{mlir::getAffineDimExpr(
                                                            x1, context),
                                                        mlir::getAffineDimExpr(
                                                            x2, context)}
                  : llvm::SmallVector<mlir::AffineExpr>{mlir::getAffineDimExpr(
                                                            x2, context),
                                                        mlir::getAffineDimExpr(
                                                            x1, context)},
      output0_affine_exprs = {mlir::getAffineDimExpr(x0, context),
                              mlir::getAffineDimExpr(x1, context)},
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
  mlir::AffineMap lhs_affine_map =
                      mlir::AffineMap::get(3, 0, lhs_affine_exprs, context),
                  rhs_affine_map =
                      mlir::AffineMap::get(3, 0, rhs_affine_exprs, context),
                  output0_affine_map =
                      mlir::AffineMap::get(3, 0, output0_affine_exprs, context),
                  bias_affine_map =
                      mlir::AffineMap::get(2, 0, bias_affine_exprs, context),
                  output1_affine_map =
                      mlir::AffineMap::get(2, 0, output1_affine_exprs, context);
  llvm::SmallVector<mlir::utils::IteratorType> matmul_iterator_types(
      3, mlir::utils::IteratorType::parallel),
      add_iterator_types(2, mlir::utils::IteratorType::parallel);
  const size_t k_index = std::distance(
      axes_.begin(), std::find(axes_.begin(), axes_.end(), Axis::k));
#ifdef DEBUG
  assert(k_index < 3);
#endif
  matmul_iterator_types[k_index] = mlir::utils::IteratorType::reduction;
  builder.create<mlir::linalg::FillOp>(builder.getUnknownLoc(), c0f, output);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{output},
      llvm::ArrayRef{lhs_affine_map, rhs_affine_map, output0_affine_map},
      matmul_iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value lhs = inputs[0], rhs = inputs[1], output = inputs[2],
                    mul_op = b.create<mlir::arith::MulFOp>(loc, lhs, rhs),
                    add_op = b.create<mlir::arith::AddFOp>(loc, mul_op, output);
        b.create<mlir::linalg::YieldOp>(loc, add_op);
      });
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{output, bias_memref}, mlir::ValueRange{output},
      llvm::ArrayRef{output1_affine_map, bias_affine_map, output1_affine_map},
      add_iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value output = inputs[0], bias = inputs[1],
                    mul0_op = b.create<mlir::arith::MulFOp>(loc, alpha, bias),
                    mul1_op = b.create<mlir::arith::MulFOp>(loc, beta, output),
                    add_op =
                        b.create<mlir::arith::AddFOp>(loc, mul0_op, mul1_op);
        b.create<mlir::linalg::YieldOp>(loc, add_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers
