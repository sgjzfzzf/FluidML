#include "structure/kernel/kernel/add.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "structure/kernel/kernel/utils.h"
#include "structure/tensor/meta.h"
#include "utils/float.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace kernel {

AddConstantKernel::AddConstantKernel(Type type, float64_t constant)
    : type_(type), constant_(constant) {}

std::string AddConstantKernel::GetKernelName() const { return kKernelName; }

void AddConstantKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                            mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Value constant;
  if (type_ == Type::kInt32) {
    constant = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(constant_));
  } else if (type_ == Type::kInt64) {
    constant = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI64IntegerAttr(constant_));
  } else if (type_ == Type::kFloat32) {
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
#endif
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output},
      llvm::SmallVector(2, builder.getMultiDimIdentityMap(rank)),
      iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0], add_op;
        mlir::Type input_elem_type = input_type.getElementType(),
                   output_elem_type = output_type.getElementType(),
                   constant_type = constant.getType();
        if (input_elem_type.isa<mlir::IntegerType>() &&
            constant_type.isa<mlir::IntegerType>() &&
            output_elem_type.isa<mlir::IntegerType>()) {
          add_op = b.create<mlir::arith::AddIOp>(loc, input, constant);
        } else if (input_elem_type.isa<mlir::FloatType>() &&
                   constant_type.isa<mlir::FloatType>() &&
                   output_elem_type.isa<mlir::FloatType>()) {
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

std::string AddCommonKernel::GetKernelName() const { return kKernelName; }

void AddCommonKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                          mlir::Value &rhs, mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType lhs_type = mlir::cast<mlir::MemRefType>(lhs.getType()),
                   rhs_type = mlir::cast<mlir::MemRefType>(rhs.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = output_type.getRank();
  Type lhs_raw_type = GetType(lhs_type.getElementType()),
       rhs_raw_type = GetType(rhs_type.getElementType()),
       output_raw_type = GetType(output_type.getElementType());
#ifdef DEBUG
  assert(lhs_type.getRank() <= rank);
  assert(rhs_type.getRank() <= rank);
  Meta lhs_meta(lhs_raw_type, lhs_type.getShape()),
      rhs_meta(rhs_raw_type, rhs_type.getShape()),
      output_meta(output_raw_type, output_type.getShape());
  std::optional<Meta> broadcast_meta_opt =
      BroadcastShape(lhs_meta, rhs_meta, output_raw_type);
  assert(broadcast_meta_opt.has_value());
  assert(*broadcast_meta_opt == output_meta);
#endif
  llvm::SmallVector<mlir::AffineMap> maps =
      GetBroadcastAffineMaps(builder, {lhs_type, rhs_type}, output_type);
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
        mlir::Value lhs = inputs[0], rhs = inputs[1], add_op;
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
