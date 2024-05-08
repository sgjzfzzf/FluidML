#include "structure/kernel/mul.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#ifdef DEBUG
#include "exception/unreachable_exception.h"
#endif

namespace cpu_transformers {
namespace kernel {

MulConstantScalarKernel::MulConstantScalarKernel(Type type, float64_t constant)
    : type_(type), constant_(constant) {}

void MulConstantScalarKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                                  mlir::Value &output) {
  mlir::Value constant;
  if (type_ == Type::FLOAT32) {
    constant = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getF32FloatAttr(constant_));
  } else {
#ifdef DEBUG
    throw UnreachableException();
#else
    __builtin_unreachable();
#endif
  }
  mlir::MemRefType lhs_type = mlir::cast<mlir::MemRefType>(lhs.getType());
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  size_t rank = lhs_type.getRank();
#ifdef DEBUG
  assert(rank == output_type.getRank());
#endif
  mlir::AffineMap lhs_map = builder.getMultiDimIdentityMap(rank);
  mlir::AffineMap output_map = builder.getMultiDimIdentityMap(rank);
  llvm::SmallVector<mlir::AffineMap> maps = {lhs_map, output_map};
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{lhs},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value lhs = inputs[0];
        mlir::Value mul_op = b.create<mlir::arith::MulFOp>(loc, lhs, constant);
        b.create<mlir::linalg::YieldOp>(loc, mul_op);
      });
}

MulConstantTensorKernel::MulConstantTensorKernel(const Tensor &constant)
    : constant_(constant) {}

void MulConstantTensorKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                                  mlir::Value &output) {
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType());
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  size_t rank = output_type.getRank();
  const std::vector<int64_t> &input_shape = input_type.getShape();
  const std::vector<int64_t> &output_shape = output_type.getShape();
  const std::vector<int64_t> &weight_shape = constant_.GetShape();
  const std::vector<float64_t> &weight_ref = constant_.Get();
#ifdef DEBUG
  assert(rank >= input_shape.size());
  assert(rank >= weight_shape.size());
  assert(rank == output_shape.size());
#endif
  mlir::Type weight_type = GetMLIRType(constant_.GetType(), builder);
  mlir::RankedTensorType weight_tensor_type =
      mlir::RankedTensorType::get(weight_shape, weight_type);
  mlir::DenseElementsAttr weight_elements;
  if (constant_.GetType() == Type::FLOAT32) {
    std::vector<float32_t> weight_data(weight_ref.begin(), weight_ref.end());
    weight_elements = mlir::DenseElementsAttr::get(
        weight_tensor_type, llvm::ArrayRef<float32_t>(weight_data));
  } else {
#ifdef DEBUG
    throw UnreachableException();
#else
    __builtin_unreachable();
#endif
  }
  mlir::Value constant_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), weight_elements);
  mlir::MemRefType constant_memref_type =
      mlir::MemRefType::get(weight_shape, weight_type);
  mlir::Value constant_buffer = builder.create<mlir::bufferization::ToMemrefOp>(
      builder.getUnknownLoc(), constant_memref_type, constant_value);
  llvm::SmallVector<mlir::AffineMap> maps =
      getBroadcastAffineMaps(builder,
                             {
                                 input_type,
                                 constant_memref_type,
                             },
                             output_type);
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{input, constant_buffer}, mlir::ValueRange{output}, maps,
      iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value lhs = inputs[0], weight = inputs[1];
        mlir::Value mul_op = b.create<mlir::arith::MulFOp>(loc, lhs, weight);
        b.create<mlir::linalg::YieldOp>(loc, mul_op);
      });
}

void MulCommonKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                          mlir::Value &rhs, mlir::Value &output) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType lhs_type = mlir::cast<mlir::MemRefType>(lhs.getType());
  mlir::MemRefType rhs_type = mlir::cast<mlir::MemRefType>(rhs.getType());
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = output_type.getRank();
#ifdef DEBUG
  assert(rank <= lhs_type.getRank());
  assert(rank <= rhs_type.getRank());
#endif
  llvm::SmallVector<mlir::AffineMap> maps =
      getBroadcastAffineMaps(builder, {lhs_type, rhs_type}, output_type);
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value lhs = inputs[0], rhs = inputs[1];
        mlir::Value mul_op = b.create<mlir::arith::MulFOp>(loc, lhs, rhs);
        b.create<mlir::linalg::YieldOp>(loc, mul_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers
