#include "structure/kernel/kernel/reduce_mean.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"

namespace fluidml {
namespace kernel {

ReduceMeanKernel::ReduceMeanKernel(llvm::SmallVector<int64_t> &&axes,
                                   bool keep_dims)
    : axes_(std::move(axes)), keep_dims_(keep_dims) {}

std::string ReduceMeanKernel::GetKernelName() const { return kKernelName; }

void ReduceMeanKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                           mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t input_rank = input_type.getRank(),
               output_rank = output_type.getRank(), axes_len = axes_.size();
  llvm::ArrayRef input_shape = input_type.getShape(),
                 output_shape = output_type.getShape();
  mlir::Type element_type = input_type.getElementType();
#ifdef DEBUG
  if (keep_dims_) {
    assert(input_rank == output_rank);
  } else {
    assert(input_rank == output_rank + axes_len);
  }
  assert(element_type == output_type.getElementType());
#endif
  int64_t divisor = 1;
  for (int64_t axis : axes_) {
#ifdef DEBUG
    assert(axis >= 0 && axis < input_rank);
    if (keep_dims_) {
      assert(output_shape[axis] == 1);
    }
#endif
    divisor *= input_shape[axis];
  }
  llvm::SmallVector<mlir::AffineExpr> output_exprs;
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types;
  if (keep_dims_) {
    for (size_t i = 0; i < output_rank; ++i) {
      output_exprs.push_back(builder.getAffineDimExpr(i));
    }
    iterator_types = llvm::SmallVector<mlir::utils::IteratorType>(
        input_rank, mlir::utils::IteratorType::parallel);
    for (int64_t axis : axes_) {
      output_exprs[axis] = builder.getAffineConstantExpr(0);
      iterator_types[axis] = mlir::utils::IteratorType::reduction;
    }
  } else {
#ifdef DEBUG
    assert(false && "unimplemented");
#else
    __builtin_unreachable();
#endif
  }
  mlir::AffineMap output_map =
      mlir::AffineMap::get(output_rank, 0, output_exprs, context);
  mlir::Value c0 = builder.create<mlir::arith::ConstantOp>(
                  builder.getUnknownLoc(),
                  builder.getFloatAttr(element_type, 0.0)),
              mul_op = builder.create<mlir::arith::ConstantOp>(
                  builder.getUnknownLoc(),
                  builder.getFloatAttr(element_type, divisor));
  builder.create<mlir::linalg::FillOp>(builder.getUnknownLoc(), c0, output);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output},
      llvm::ArrayRef{builder.getMultiDimIdentityMap(input_rank), output_map},
      iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0], output = inputs[1],
                    add_op = b.create<mlir::arith::AddFOp>(loc, input, output);
        b.create<mlir::linalg::YieldOp>(loc, add_op);
      });
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{output},
      mlir::ValueRange{output},
      llvm::SmallVector(2, builder.getMultiDimIdentityMap(output_rank)),
      llvm::SmallVector(output_rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0],
                    div_op = b.create<mlir::arith::DivFOp>(loc, input, mul_op);
        b.create<mlir::linalg::YieldOp>(loc, div_op);
      });
}

} // namespace kernel
} // namespace fluidml
