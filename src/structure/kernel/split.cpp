#include "structure/kernel/split.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include <cstdint>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

SplitKernel::SplitKernel(int64_t axis) : axis_(axis) {}

std::string SplitKernel::GetKernelName() const { return kKernelName; }

void SplitKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                      mlir::ValueRange outputs) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType());
  llvm::ArrayRef<int64_t> input_shape = input_type.getShape();
  size_t offset = 0;
  for (mlir::Value output : outputs) {
    mlir::MemRefType output_type =
        mlir::cast<mlir::MemRefType>(output.getType());
    llvm::ArrayRef<int64_t> shape = output_type.getShape();
    int64_t rank = input_type.getRank();
#ifdef DEBUG
    assert(input_type.getRank() == output_type.getRank());
    assert(input_shape.size() == shape.size());
#endif
    llvm::SmallVector<int64_t> offsets(rank, 0), strides(rank, 1);
    offsets[axis_] = offset;
    offset += shape[axis_];
    mlir::memref::SubViewOp view = builder.create<mlir::memref::SubViewOp>(
        builder.getUnknownLoc(), input, offsets, shape, strides);
    builder.create<mlir::memref::CopyOp>(builder.getUnknownLoc(), view, output);
  }
#ifdef DEBUG
  assert(offset == input_shape[axis_]);
#endif
}

} // namespace kernel
} // namespace cpu_transformers
