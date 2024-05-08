#include "structure/kernel/reshape.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#ifdef DEBUG
#include <cassert>
#include <numeric>
#endif

namespace cpu_transformers {
namespace kernel {
void ReshapeKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                        mlir::Value &output) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType());
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  llvm::ArrayRef<int64_t> input_shape = input_type.getShape();
  llvm::ArrayRef<int64_t> output_shape = output_type.getShape();
#ifdef DEBUG
  assert(std::accumulate(input_shape.begin(), input_shape.end(), 1,
                         std::multiplies<int64_t>()) ==
         std::accumulate(output_shape.begin(), output_shape.end(), 1,
                         std::multiplies<int64_t>()));
#endif
  mlir::RankedTensorType shape_tensor_type = mlir::RankedTensorType::get(
      {output_type.getRank()}, builder.getIndexType());
  mlir::Value shape = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), shape_tensor_type,
      builder.getIndexTensorAttr(output_shape));
  mlir::MemRefType shape_type =
      mlir::MemRefType::get(output_type.getRank(), builder.getIndexType());
  mlir::bufferization::ToMemrefOp shape_buffer =
      builder.create<mlir::bufferization::ToMemrefOp>(builder.getUnknownLoc(),
                                                      shape_type, shape);
  mlir::memref::ReshapeOp reshape_view =
      builder.create<mlir::memref::ReshapeOp>(builder.getUnknownLoc(),
                                              output_type, input, shape_buffer);
  builder.create<mlir::memref::CopyOp>(builder.getUnknownLoc(), reshape_view,
                                       output);
}
} // namespace kernel
} // namespace cpu_transformers
