#include "structure/kernel/kernel/pow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"

namespace fluidml {
namespace kernel {

PowKernel::PowKernel(Type type, float64_t exp) : type_(type), exp_(exp) {}

std::string PowKernel::GetKernelName() const { return kKernelName; }

void PowKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                    mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Value exp;
  if (type_ == Type::kFloat32) {
    exp = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getF32FloatAttr(exp_));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = input_type.getRank();
#ifdef DEBUG
  assert(rank == output_type.getRank());
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
        mlir::Value input = inputs[0], output = inputs[1], pow_op;
        mlir::Type input_type = input.getType(), exp_type = exp.getType(),
                   output_type = output.getType();
        if (input_type.isa<mlir::FloatType>() &&
            exp_type.isa<mlir::FloatType>() &&
            output_type.isa<mlir::FloatType>()) {
          pow_op = b.create<mlir::math::PowFOp>(loc, input, exp);
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
        b.create<mlir::linalg::YieldOp>(loc, pow_op);
      });
}

} // namespace kernel
} // namespace fluidml
