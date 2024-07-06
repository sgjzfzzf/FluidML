#include "structure/kernel/unsqueeze_sub_mul.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace cpu_transformers {
namespace kernel {

UnsqueezeSubLhsScalarMulRhsScalarKernel::
    UnsqueezeSubLhsScalarMulRhsScalarKernel(
        llvm::ArrayRef<int64_t> unsqueeze_axes, const Type &sub_type,
        float64_t sub_val, const Type &mul_type, float64_t mul_val)
    : unsqueeze_axes_(unsqueeze_axes), sub_type_(sub_type), sub_val_(sub_val),
      mul_type_(mul_type), mul_val_(mul_val) {}

void UnsqueezeSubLhsScalarMulRhsScalarKernel::Run(mlir::OpBuilder &builder,
                                                  mlir::Value &input,
                                                  mlir::Value &output) const {
// TODO: Only support for fp32 currently
#ifdef DEBUG
  assert(sub_type_ == Type::FLOAT32);
  assert(mul_type_ == Type::FLOAT32);
#endif
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t output_rank = output_type.getRank();
#ifdef DEBUG
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType());
  const size_t input_rank = input_type.getRank();
  assert(input_rank + unsqueeze_axes_.size() == output_rank);
#endif
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (size_t i = 0; i < output_rank; ++i) {
    bool is_axis = false;
    for (int64_t axis : unsqueeze_axes_) {
      if (i == axis) {
        is_axis = true;
        break;
      }
    }
    if (!is_axis) {
      exprs.push_back(builder.getAffineDimExpr(i));
    }
  }
  mlir::AffineMap input_map =
                      mlir::AffineMap::get(output_rank, 0, exprs, context),
                  output_map = builder.getMultiDimIdentityMap(output_rank);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output},
      llvm::ArrayRef<mlir::AffineMap>{input_map, output_map},
      llvm::SmallVector<mlir::utils::IteratorType>(
          output_rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0];
        mlir::Value sub = b.create<mlir::arith::ConstantOp>(
            loc, builder.getF32FloatAttr(sub_val_));
        mlir::Value mul = b.create<mlir::arith::ConstantOp>(
            loc, builder.getF32FloatAttr(mul_val_));
        mlir::Value sub_op = b.create<mlir::arith::SubFOp>(loc, sub, input);
        mlir::Value mul_op = b.create<mlir::arith::MulFOp>(loc, sub_op, mul);
        b.create<mlir::linalg::YieldOp>(loc, mul_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers
