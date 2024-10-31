#include "structure/kernel/kernel/neg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#ifdef DEBUG
#include <cassert>
#endif

namespace fluidml {
namespace kernel {

std::string NegKernel::GetKernelName() const { return kKernelName; }

void NegKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
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
        mlir::Value input = inputs[0],
                    neg_op = b.create<mlir::arith::NegFOp>(loc, input);
        b.create<mlir::linalg::YieldOp>(loc, neg_op);
      });
}

} // namespace kernel
} // namespace fluidml
