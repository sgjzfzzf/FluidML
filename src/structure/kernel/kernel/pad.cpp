#include "structure/kernel/kernel/pad.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace fluidml {
namespace kernel {

PadKernel::PadKernel(std::vector<std::tuple<int64_t, int64_t>> &&pads)
    : pads_(std::move(pads)) {}

std::string PadKernel::GetKernelName() const { return kKernelName; }

void PadKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                    mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = pads_.size();
  llvm::ArrayRef input_shape = input_type.getShape(),
                 output_shape = output_type.getShape();
#ifdef DEBUG
  assert(rank == input_shape.size());
  assert(rank == output_shape.size());
  for (size_t i = 0; i < rank; ++i) {
    const auto &[pad_before, pad_after] = pads_[i];
    assert(input_shape[i] + pad_before + pad_after == output_shape[i]);
  }
  assert(input_type.getElementType() == output_type.getElementType());
#endif
  llvm::SmallVector<int64_t> offsets;
  for (const auto &[offset, _] : pads_) {
    offsets.push_back(offset);
  }
  mlir::Value subview_op = builder.create<mlir::memref::SubViewOp>(
      builder.getUnknownLoc(), output, offsets, input_shape,
      llvm::SmallVector<int64_t>(rank, 1));
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{subview_op},
      llvm::SmallVector(2, builder.getMultiDimIdentityMap(rank)),
      llvm::SmallVector(rank, mlir::utils::IteratorType::parallel),
      [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0];
        b.create<mlir::linalg::YieldOp>(loc, input);
      });
}

} // namespace kernel
} // namespace fluidml
