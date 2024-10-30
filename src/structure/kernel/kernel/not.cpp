#include "structure/kernel/kernel/not.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"

namespace cpu_transformers {
namespace kernel {

std::string NotKernel::GetKernelName() const { return kKernelName; }

void NotKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                    mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Type i1_type = builder.getI1Type();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
#ifdef DEBUG
  assert(input_type.getElementType() == i1_type);
  assert(output_type.getElementType() == i1_type);
#endif
  size_t rank = input_type.getRank();
#ifdef DEBUG
  assert(rank == output_type.getRank());
#endif
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output},
      llvm::SmallVector(2, builder.getMultiDimIdentityMap(rank)),
      llvm::SmallVector<mlir::utils::IteratorType>(
          rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0],
                    true_op = b.create<mlir::arith::ConstantOp>(
                        loc, i1_type, b.getIntegerAttr(i1_type, 1)),
                    xor_op = b.create<mlir::arith::XOrIOp>(loc, input, true_op);
        b.create<mlir::linalg::YieldOp>(loc, xor_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers