#include "structure/kernel/kernel/equal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

namespace fluidml {
namespace kernel {

EqualKernel::EqualKernel(Type type, float64_t value)
    : type_(type), value_(value) {}

std::string EqualKernel::GetKernelName() const { return kKernelName; }

void EqualKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                      mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Value value;
  if (type_ == Type::kInt64) {
    value = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI64IntegerAttr(value_));
  } else {
#ifdef DEBUG
    assert(false && "unimplemented");
#else
    __builtin_unreachable();
#endif
  }
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
#ifdef DEBUG
  assert(output_type.getElementType().isSignlessInteger(1));
#endif
  const size_t rank = input_type.getRank();
#ifdef DEBUG
  assert(rank == output_type.getRank());
#endif
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output},
      llvm::SmallVector(2, builder.getMultiDimIdentityMap(rank)),
      llvm::SmallVector(2, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0],
                    equal_op = b.create<mlir::arith::CmpIOp>(
                        loc, mlir::arith::CmpIPredicate::eq, input, value);
        b.create<mlir::linalg::YieldOp>(loc, equal_op);
      });
}

} // namespace kernel
} // namespace fluidml