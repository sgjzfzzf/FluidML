#include "structure/kernel/kernel/sub.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "structure/kernel/kernel/utils.h"

namespace cpu_transformers {
namespace kernel {

SubConstantLhsKernel::SubConstantLhsKernel(Type type, float64_t value)
    : type_(type), value_(value) {}

std::string SubConstantLhsKernel::GetKernelName() const { return kKernelName; }

void SubConstantLhsKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                               mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Value value;
  if (type_ == Type::kFloat32) {
    value = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getF32FloatAttr(value_));
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
  mlir::AffineMap input_map = builder.getMultiDimIdentityMap(rank),
                  output_map = builder.getMultiDimIdentityMap(rank);
  llvm::SmallVector<mlir::AffineMap> maps = {input_map, output_map};
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types;
  for (size_t i = 0; i < rank; ++i) {
    iterator_types.push_back(mlir::utils::IteratorType::parallel);
  }
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0], output = inputs[1], sub_op;
        mlir::Type input_type = input.getType(), value_type = value.getType(),
                   output_type = output.getType();
        if (input_type.isa<mlir::FloatType>() &&
            value_type.isa<mlir::FloatType>() &&
            output_type.isa<mlir::FloatType>()) {
          sub_op = b.create<mlir::arith::SubFOp>(loc, value, input);
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
        b.create<mlir::linalg::YieldOp>(loc, sub_op);
      });
}

std::string SubCommonKernel::GetKernelName() const { return kKernelName; }

void SubCommonKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                          mlir::Value &rhs, mlir::Value &output) const {
  mlir::MemRefType lhs_type = mlir::cast<mlir::MemRefType>(lhs.getType()),
                   rhs_type = mlir::cast<mlir::MemRefType>(rhs.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  size_t rank = lhs_type.getRank();
#ifdef DEBUG
  assert(rank == rhs_type.getRank());
  assert(rank == output_type.getRank());
#endif
  llvm::ArrayRef lhs_shape = lhs_type.getShape(),
                 rhs_shape = rhs_type.getShape(),
                 output_shape = output_type.getShape();
  llvm::SmallVector<mlir::AffineMap> maps =
      GetBroadcastAffineMaps(builder, {lhs_type, rhs_type}, output_type);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{output}, maps,
      llvm::SmallVector<mlir::utils::IteratorType>(
          rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value lhs = inputs[0], rhs = inputs[1], output = inputs[2],
                    sub_op = b.create<mlir::arith::SubFOp>(loc, lhs, rhs);
        b.create<mlir::linalg::YieldOp>(loc, sub_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers
