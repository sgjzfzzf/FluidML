#include "structure/kernel/kernel/dropout.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace cpu_transformers {
namespace kernel {

DropoutKernel::DropoutKernel(float64_t ratio) : ratio_(ratio) {}

std::string DropoutKernel::GetKernelName() const { return kKernelName; }

void DropoutKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
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
      mlir::SmallVector(2, builder.getMultiDimIdentityMap(rank)),
      llvm::SmallVector(rank, mlir::utils::IteratorType::parallel),
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
