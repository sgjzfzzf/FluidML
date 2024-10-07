#include "structure/kernel/sub.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"

namespace cpu_transformers {
namespace kernel {

SubConstantScalarLhsKernel::SubConstantScalarLhsKernel(Type type,
                                                       float64_t value)
    : type_(type), value_(value) {}

void SubConstantScalarLhsKernel::Run(mlir::OpBuilder &builder,
                                     mlir::Value &input,
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
  Type input_raw_type = GetType(input_type.getElementType()),
       output_raw_type = GetType(output_type.getElementType());
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
        if (input_raw_type == Type::kFloat32 && type_ == Type::kFloat32 &&
            output_raw_type == Type::kFloat32) {
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

} // namespace kernel
} // namespace cpu_transformers
