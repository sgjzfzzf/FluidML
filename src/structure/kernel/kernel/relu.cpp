#include "structure/kernel/kernel/relu.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"

namespace fluidml {
namespace kernel {

std::string ReluKernel::GetKernelName() const { return kKernelName; }

void ReluKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                     mlir::Value &output) const {
  mlir::MemRefType input_type = input.getType().cast<mlir::MemRefType>(),
                   output_type = output.getType().cast<mlir::MemRefType>();
  const size_t rank = input_type.getRank();
#ifdef DEBUG
  assert(input_type.getShape() == output_type.getShape());
#endif
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output},
      llvm::SmallVector(2, builder.getMultiDimIdentityMap(rank)),
      llvm::SmallVector<mlir::utils::IteratorType>(
          rank, mlir::utils::IteratorType::parallel),
      [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0], output = inputs[1], relu_op;
        mlir::Type input_type = input.getType(), output_type = output.getType();
        if (input_type.isa<mlir::FloatType>() &&
            output_type.isa<mlir::FloatType>()) {
          mlir::Value zero_op = b.create<mlir::arith::ConstantOp>(
              loc, b.getFloatAttr(input_type, 0.0));
          relu_op = b.create<mlir::arith::MaximumFOp>(loc, input, zero_op);
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
        b.create<mlir::linalg::YieldOp>(loc, relu_op);
      });
}

} // namespace kernel
} // namespace fluidml
