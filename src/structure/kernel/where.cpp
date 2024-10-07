#include "structure/kernel/where.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "structure/tensor/tensor.h"
#include "utils/float.h"
#include "utils/type.h"
#include <cstdint>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

WhereConstantCondConstantScalarYKernel::WhereConstantCondConstantScalarYKernel(
    Tensor &&cond, Type type, float64_t y)
    : cond_(std::move(cond)), type_(type), y_(y) {}

void WhereConstantCondConstantScalarYKernel::Run(mlir::OpBuilder &builder,
                                                 mlir::Value &input,
                                                 mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  const std::vector<int64_t> &cond_shape = cond_.GetShape();
  const std::vector<float64_t> &cond_ref = cond_.Get();
  mlir::MemRefType x_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = output_type.getRank();
#ifdef DEBUG
  assert(cond_.GetType() == Type::kBool);
  // TODO: add support for other types in the future
  assert(type_ == Type::kFloat32);
#endif
  mlir::RankedTensorType cond_tensor_type =
      mlir::RankedTensorType::get(cond_shape, builder.getI1Type());
  mlir::MemRefType cond_memref_type =
      mlir::MemRefType::get(cond_shape, builder.getI1Type());
  llvm::SmallVector<mlir::APInt> cond_data;
  for (bool i : cond_ref) {
    cond_data.push_back(mlir::APInt(1, i, true));
  }
  mlir::DenseElementsAttr cond_elements =
      mlir::DenseElementsAttr::get(cond_tensor_type, cond_data);
  mlir::arith::ConstantOp cond_value = builder.create<mlir::arith::ConstantOp>(
                              builder.getUnknownLoc(), cond_elements),
                          y_value = builder.create<mlir::arith::ConstantOp>(
                              builder.getUnknownLoc(),
                              builder.getF32FloatAttr(y_));
  mlir::bufferization::ToMemrefOp cond_memref =
      builder.create<mlir::bufferization::ToMemrefOp>(
          builder.getUnknownLoc(), cond_memref_type, cond_value);
  llvm::SmallVector<mlir::AffineMap> maps = getBroadcastAffineMaps(
      builder, llvm::ArrayRef<mlir::MemRefType>{cond_memref_type, x_type},
      output_type);
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{cond_memref, input}, mlir::ValueRange{output}, maps,
      iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value select = inputs[0], x = inputs[1],
                    select_op = b.create<mlir::arith::SelectOp>(loc, select, x,
                                                                y_value);
        b.create<mlir::linalg::YieldOp>(loc, select_op);
      });
}

WhereConstantCondConstantTensorYKernel::WhereConstantCondConstantTensorYKernel(
    Tensor &&cond, Tensor &&y)
    : cond_(std::move(cond)), y_(std::move(y)) {}

void WhereConstantCondConstantTensorYKernel::Run(mlir::OpBuilder &builder,
                                                 mlir::Value &input,
                                                 mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  const std::vector<int64_t> &cond_shape = cond_.GetShape(),
                             &y_shape = y_.GetShape();
  const std::vector<float64_t> &cond_ref = cond_.Get(), &y_ref = y_.Get();
  mlir::MemRefType x_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = output_type.getRank();
#ifdef DEBUG
  assert(cond_.GetType() == Type::kBool);
  // TODO: add support for other types in the future
  assert(y_.GetType() == Type::kFloat32);
#endif
  llvm::SmallVector<mlir::APInt> cond_data;
  for (bool i : cond_ref) {
    cond_data.push_back(mlir::APInt(1, i, true));
  }
  mlir::RankedTensorType cond_tensor_type =
      mlir::RankedTensorType::get(cond_shape, builder.getI1Type());
  mlir::MemRefType cond_memref_type =
      mlir::MemRefType::get(cond_shape, builder.getI1Type());
  mlir::DenseElementsAttr cond_elements =
      mlir::DenseElementsAttr::get(cond_tensor_type, cond_data);
  mlir::arith::ConstantOp cond_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), cond_elements);
  mlir::bufferization::ToMemrefOp cond_memref =
      builder.create<mlir::bufferization::ToMemrefOp>(
          builder.getUnknownLoc(), cond_memref_type, cond_value);
  llvm::SmallVector<float32_t> y_data(y_ref.begin(), y_ref.end());
  mlir::RankedTensorType y_tensor_type =
      mlir::RankedTensorType::get(y_shape, builder.getF32Type());
  mlir::MemRefType y_memref_type =
      mlir::MemRefType::get(y_shape, builder.getF32Type());
  mlir::DenseElementsAttr y_elements =
      mlir::DenseElementsAttr::get(y_tensor_type, llvm::ArrayRef(y_data));
  mlir::arith::ConstantOp y_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), y_elements);
  mlir::bufferization::ToMemrefOp y_memref =
      builder.create<mlir::bufferization::ToMemrefOp>(builder.getUnknownLoc(),
                                                      y_memref_type, y_value);
  llvm::SmallVector<mlir::AffineMap> maps = getBroadcastAffineMaps(
      builder, llvm::ArrayRef<mlir::MemRefType>{cond_memref_type, x_type},
      output_type);
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{cond_memref, input, y_memref}, mlir::ValueRange{output},
      maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 4);
#endif
        mlir::Value select = inputs[0], x = inputs[1], y = inputs[2],
                    select_op =
                        b.create<mlir::arith::SelectOp>(loc, select, x, y);
        b.create<mlir::linalg::YieldOp>(loc, select_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers
