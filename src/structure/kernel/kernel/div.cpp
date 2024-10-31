#include "structure/kernel/kernel/div.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "structure/kernel/kernel/utils.h"
#include "utils/type.h"

namespace fluidml {
namespace kernel {

DivConstantRhsKernel::DivConstantRhsKernel(Type type, float64_t constant)
    : type_(type), constant_(constant) {}

std::string DivConstantRhsKernel::GetKernelName() const { return kKernelName; }

void DivConstantRhsKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                               mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Value constant;
  if (type_ == Type::kInt32) {
    constant = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(constant_));
  }
  if (type_ == Type::kFloat32) {
    constant = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getF32FloatAttr(constant_));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  size_t rank = input_type.getRank();
#ifdef DEBUG
  assert(rank == output_type.getRank());
  assert(input_type.getShape() == output_type.getShape());
#endif
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output},
      llvm::SmallVector(2, builder.getMultiDimIdentityMap(rank)),
      llvm::SmallVector(rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0], output = inputs[1], div_op;
        if (input.getType().isa<mlir::FloatType>() &&
            constant.getType().isa<mlir::FloatType>() &&
            output.getType().isa<mlir::FloatType>()) {
          div_op = b.create<mlir::arith::DivFOp>(loc, input, constant);
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
        b.create<mlir::linalg::YieldOp>(loc, div_op);
      });
}

std::string DivCommonKernel::GetKernelName() const { return kKernelName; }

void DivCommonKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                          mlir::Value &rhs, mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType lhs_type = mlir::cast<mlir::MemRefType>(lhs.getType()),
                   rhs_type = mlir::cast<mlir::MemRefType>(rhs.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  size_t rank = lhs_type.getRank();
#ifdef DEBUG
  assert(rank == rhs_type.getRank());
  assert(rank == output_type.getRank());
#endif
  llvm::SmallVector<mlir::AffineMap> maps =
      GetBroadcastAffineMaps(builder, {lhs_type, rhs_type}, output_type);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{output}, maps,
      llvm::SmallVector(rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value lhs = inputs[0], rhs = inputs[1], output = inputs[2],
                    div_op;
        if (lhs.getType().isF32() && rhs.getType().isF32() &&
            output.getType().isF32()) {
          div_op = b.create<mlir::arith::DivFOp>(loc, lhs, rhs);
        } else {
#ifdef DEBUG
          assert(false && "unimplemented");
#else
          __builtin_unreachable();
#endif
        }
        b.create<mlir::linalg::YieldOp>(loc, div_op);
      });
}

} // namespace kernel
} // namespace fluidml
