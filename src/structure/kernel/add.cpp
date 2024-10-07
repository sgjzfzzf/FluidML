#include "structure/kernel/add.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "structure/tensor/meta.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

AddConstantScalarKernel::AddConstantScalarKernel(Type type, float64_t constant)
    : type_(type), constant_(constant) {}

void AddConstantScalarKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                                  mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Value constant;
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
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType());
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  size_t rank = input_type.getRank();
#ifdef DEBUG
  assert(rank == output_type.getRank());
#endif
  Type input_raw_type = GetType(input_type.getElementType());
  Type output_raw_type = GetType(output_type.getElementType());
  mlir::AffineMap input_map = builder.getMultiDimIdentityMap(rank);
  mlir::AffineMap output_map = builder.getMultiDimIdentityMap(rank);
  llvm::SmallVector<mlir::AffineMap> maps = {input_map, output_map};
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0];
        mlir::Value add_op;
        if (input_raw_type == Type::kFloat32 &&
            output_raw_type == Type::kFloat32 && type_ == Type::kFloat32) {
          add_op = b.create<mlir::arith::AddFOp>(loc, input, constant);
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
        b.create<mlir::linalg::YieldOp>(loc, add_op);
      });
}

void AddCommonKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                          mlir::Value &rhs, mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType lhs_type = mlir::cast<mlir::MemRefType>(lhs.getType());
  mlir::MemRefType rhs_type = mlir::cast<mlir::MemRefType>(rhs.getType());
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  size_t rank = output_type.getRank();
  Type lhs_raw_type = GetType(lhs_type.getElementType());
  Type rhs_raw_type = GetType(rhs_type.getElementType());
  Type output_raw_type = GetType(output_type.getElementType());
#ifdef DEBUG
  assert(lhs_type.getRank() <= rank);
  assert(rhs_type.getRank() <= rank);
  Meta lhs_meta(lhs_raw_type, lhs_type.getShape());
  Meta rhs_meta(rhs_raw_type, rhs_type.getShape());
  Meta output_meta(output_raw_type, output_type.getShape());
  std::optional<Meta> broadcast_meta_opt =
      BroadcastShape(lhs_meta, rhs_meta, output_raw_type);
  assert(broadcast_meta_opt.has_value());
  assert(*broadcast_meta_opt == output_meta);
#endif
  llvm::SmallVector<mlir::AffineMap> maps =
      getBroadcastAffineMaps(builder, {lhs_type, rhs_type}, output_type);
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types;
  for (size_t i = 0; i < rank; ++i) {
    iterator_types.push_back(mlir::utils::IteratorType::parallel);
  }
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value lhs = inputs[0];
        mlir::Value rhs = inputs[1];
        mlir::Value add_op;
        if (lhs_raw_type == Type::kFloat32 && rhs_raw_type == Type::kFloat32 &&
            output_raw_type == Type::kFloat32) {
          add_op = b.create<mlir::arith::AddFOp>(loc, lhs, rhs);
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
        b.create<mlir::linalg::YieldOp>(loc, add_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers
