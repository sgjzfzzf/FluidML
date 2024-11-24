#include "structure/kernel/kernel/clip.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"

namespace fluidml {
namespace kernel {

ClipKernel::ClipKernel(float32_t min, float32_t max) : min_(min), max_(max) {}

std::string ClipKernel::GetKernelName() const { return kKernelName; }

void ClipKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                     mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = input_type.getRank();
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output},
      mlir::SmallVector(2, builder.getMultiDimIdentityMap(rank)),
      llvm::SmallVector(rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0], output = inputs[1], output_op;
        mlir::Type input_type = input.getType(), output_type = output.getType();
        if (input_type.isa<mlir::FloatType>() &&
            output_type.isa<mlir::FloatType>()) {
          mlir::Value min = b.create<mlir::arith::ConstantOp>(
                          loc, builder.getFloatAttr(input_type, min_)),
                      max = b.create<mlir::arith::ConstantOp>(
                          loc, builder.getFloatAttr(input_type, max_));
          mlir::Value clipped =
              b.create<mlir::arith::MaximumFOp>(loc, input, min);
          output_op = b.create<mlir::arith::MinimumFOp>(loc, clipped, max);
        } else {
#ifdef DEBUG
          assert(false && "unimplemented");
#else
          __builtin_unreachable();
#endif
        }
        b.create<mlir::linalg::YieldOp>(loc, output_op);
      });
}

} // namespace kernel
} // namespace fluidml
