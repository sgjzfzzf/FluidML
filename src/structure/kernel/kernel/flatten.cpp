#include "structure/kernel/kernel/flatten.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include <numeric>

namespace fluidml {
namespace kernel {

FlattenKernel::FlattenKernel(int64_t axis) : axis_(axis) {}

std::string FlattenKernel::GetKernelName() const { return kKernelName; }

void FlattenKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                        mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = input.getType().cast<mlir::MemRefType>(),
                   output_type = output.getType().cast<mlir::MemRefType>();
  llvm::ArrayRef input_shape = input_type.getShape(),
                 output_shape = output_type.getShape();
  const size_t input_rank = input_shape.size(),
               output_rank = output_shape.size(),
               axis = axis_ >= 0 ? axis_ : input_rank + axis_;
#ifdef DEBUG
  assert(input_rank >= axis);
  assert(output_rank == 2);
#endif
  mlir::AffineMap input_map = builder.getMultiDimIdentityMap(input_rank);
  mlir::AffineExpr dim0 = builder.getAffineConstantExpr(0),
                   dim1 = builder.getAffineConstantExpr(0);
  size_t strides =
      std::accumulate(input_shape.begin(), input_shape.begin() + axis, 1,
                      std::multiplies<int64_t>());
  for (size_t i = 0; i < axis; ++i) {
    const int64_t dim = input_shape[i];
    strides /= dim;
    dim0 = dim0 +
           builder.getAffineDimExpr(i) * builder.getAffineConstantExpr(strides);
  }
  strides = std::accumulate(input_shape.begin() + axis, input_shape.end(), 1,
                            std::multiplies<int64_t>());
  for (size_t i = axis; i < input_rank; ++i) {
    const int64_t dim = input_shape[i];
    strides /= dim;
    dim1 = dim1 +
           builder.getAffineDimExpr(i) * builder.getAffineConstantExpr(strides);
  }
  llvm::SmallVector<mlir::AffineExpr> output_exprs = {dim0, dim1};
  mlir::AffineMap output_map =
      mlir::AffineMap::get(input_rank, 0, output_exprs, context);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, input, output,
      llvm::ArrayRef{input_map, output_map},
      llvm::SmallVector(input_rank, mlir::utils::IteratorType::reduction),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0], output = inputs[1];
        b.create<mlir::linalg::YieldOp>(loc, input);
      });
}

} // namespace kernel
} // namespace fluidml
