#include "structure/kernel/kernel/slice.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ValueRange.h"

namespace cpu_transformers {
namespace kernel {

SliceKernel::SliceKernel(
    llvm::SmallVector<llvm::SmallVector<int64_t, 4>> &&informations)
    : informations_(std::move(informations)) {}

std::string SliceKernel::GetKernelName() const { return kKernelName; }

void SliceKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                      mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  llvm::ArrayRef input_shape = input_type.getShape(),
                 output_shape = output_type.getShape();
  const size_t rank = input_type.getRank(), len = informations_.size();
#ifdef DEBUG
  assert(rank == output_type.getRank());
#endif
  llvm::SmallVector<int64_t> offsets(rank, 0), steps(rank, 1);
  for (size_t i = 0; i < len; ++i) {
    llvm::ArrayRef information = informations_[i];
    const int64_t start = information[0], axis = information[2],
                  stride = information[3];
#ifdef DEBUG
    assert(axis >= 0 && axis < rank);
#endif
    offsets[axis] = start;
    steps[axis] = stride;
  }
  mlir::Value subview_op = builder.create<mlir::memref::SubViewOp>(
      builder.getUnknownLoc(), input, offsets, output_shape, steps);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{subview_op},
      mlir::ValueRange{output},
      llvm::SmallVector<mlir::AffineMap>(
          llvm::SmallVector(2, builder.getMultiDimIdentityMap(rank))),
      llvm::SmallVector<mlir::utils::IteratorType>(
          rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0];
        b.create<mlir::linalg::YieldOp>(loc, input);
      });
}

} // namespace kernel
} // namespace cpu_transformers
