#include "structure/kernel/kernel/cast.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include <string>

namespace fluidml {
namespace kernel {

std::string CastKernel::GetKernelName() const { return kKernelName; }

void CastKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                     mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = input_type.getRank();
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
        mlir::Value input = inputs[0], output = inputs[1];
        mlir::Type input_type = input.getType(), output_type = output.getType();
        if (input_type.isSignlessInteger() && output_type.isSignlessInteger()) {
          const size_t input_bit_width = input_type.getIntOrFloatBitWidth(),
                       output_bit_width = output_type.getIntOrFloatBitWidth();
          if (input_bit_width == 1) {
            // If the input element type is boolean, due to the default values
            // of boolean is '0' and '1', so we should extend it as an unsigned
            // integer to avoid converting "true" to "-1".
            mlir::Value zext_op =
                b.create<mlir::arith::ExtUIOp>(loc, output_type, input);
            b.create<mlir::linalg::YieldOp>(loc, zext_op);
          } else if (input_bit_width < output_bit_width) {
            mlir::Value extsi_op =
                b.create<mlir::arith::ExtSIOp>(loc, output_type, input);
            b.create<mlir::linalg::YieldOp>(loc, extsi_op);
          } else if (input_bit_width > output_bit_width) {
            mlir::Value trunc_op =
                b.create<mlir::arith::TruncIOp>(loc, output_type, input);
            b.create<mlir::linalg::YieldOp>(loc, trunc_op);
          } else {
            b.create<mlir::linalg::YieldOp>(loc, input);
          }
        } else {
#ifdef DEBUG
          assert(false && "unimplemented");
#else
          __builtin_unreachable();
#endif
        }
      });
}

} // namespace kernel
} // namespace fluidml
