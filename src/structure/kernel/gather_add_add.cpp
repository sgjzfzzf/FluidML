#include "structure/kernel/gather_add_add.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include <cstdint>

namespace cpu_transformers {
namespace kernel {

GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel::
    GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel(Tensor &&data,
                                                           Tensor &&add0_weight,
                                                           Tensor &&add1_weight)
    : data_(std::move(data)), add0_weight_(std::move(add0_weight)),
      add1_weight_(std::move(add1_weight)) {}

void GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel::Run(
    mlir::OpBuilder &builder, mlir::Value &input, mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_memref_type =
                       mlir::cast<mlir::MemRefType>(input.getType()),
                   output_memref_type =
                       mlir::cast<mlir::MemRefType>(output.getType());
  const std::vector<int64_t> &data_shape = data_.GetShape(),
                             &add0_weight_shape = add0_weight_.GetShape(),
                             &add1_weight_shape = add1_weight_.GetShape();
  const std::vector<float64_t> &data_ref = data_.Get(),
                               &add0_weight_ref = add0_weight_.Get(),
                               &add1_weight_ref = add1_weight_.Get();
#ifdef DEBUG
  int64_t data_rank = data_shape.size(),
          add0_weight_rank = add0_weight_shape.size(),
          add1_weight_rank = add1_weight_shape.size();
#endif
  llvm::ArrayRef<int64_t> indices_shape = input_memref_type.getShape(),
                          output_shape = output_memref_type.getShape();
  const int64_t input_rank = input_memref_type.getRank(),
                output_rank = output_memref_type.getRank();
#ifdef DEBUG
  assert(data_rank + input_rank - 1 == output_rank);
  for (size_t i = 0; i < output_rank; ++i) {
    if (i < input_rank) {
      assert(indices_shape[i] == output_shape[i]);
    } else {
      assert(data_shape[i - input_rank + 1] == output_shape[i]);
    }
  }
  assert(llvm::ArrayRef<int64_t>(add0_weight_shape) == output_shape);
  assert(llvm::ArrayRef<int64_t>(add1_weight_shape) == output_shape);
#endif
  mlir::ShapedType data_shaped_type = mlir::RankedTensorType::get(
                       data_shape, mlir::FloatType::getF32(context)),
                   add0_weight_shaped_type = mlir::RankedTensorType::get(
                       add0_weight_shape, mlir::FloatType::getF32(context)),
                   add1_weight_shaped_type = mlir::RankedTensorType::get(
                       add1_weight_shape, mlir::FloatType::getF32(context));
  mlir::DenseElementsAttr data_elements, add0_weight_elements,
      add1_weight_elements;
  if (data_.GetType() == Type::kFloat32) {
    llvm::SmallVector<float32_t> data(data_ref.begin(), data_ref.end());
    data_elements =
        mlir::DenseElementsAttr::get(data_shaped_type, llvm::ArrayRef(data));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  if (add0_weight_.GetType() == Type::kFloat32) {
    llvm::SmallVector<float32_t> add0_weight(add0_weight_ref.begin(),
                                             add0_weight_ref.end());
    add0_weight_elements = mlir::DenseElementsAttr::get(
        add0_weight_shaped_type, llvm::ArrayRef(add0_weight));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  if (add1_weight_.GetType() == Type::kFloat32) {
    llvm::SmallVector<float32_t> add1_weight(add1_weight_ref.begin(),
                                             add1_weight_ref.end());
    add1_weight_elements = mlir::DenseElementsAttr::get(
        add1_weight_shaped_type, llvm::ArrayRef(add1_weight));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  mlir::Value data_value = builder.create<mlir::arith::ConstantOp>(
                  builder.getUnknownLoc(), data_elements),
              add0_weight_value = builder.create<mlir::arith::ConstantOp>(
                  builder.getUnknownLoc(), add0_weight_elements),
              add1_weight_value = builder.create<mlir::arith::ConstantOp>(
                  builder.getUnknownLoc(), add1_weight_elements);
  mlir::MemRefType data_memref_type = mlir::MemRefType::get(
                       data_shape, mlir::FloatType::getF32(context), {}, 0),
                   add0_weight_memref_type = mlir::MemRefType::get(
                       add0_weight_shape, mlir::FloatType::getF32(context), {},
                       0),
                   add1_weight_memref_type = mlir::MemRefType::get(
                       add1_weight_shape, mlir::FloatType::getF32(context), {},
                       0);
  mlir::Value data_memref = builder.create<mlir::bufferization::ToMemrefOp>(
                  builder.getUnknownLoc(), data_memref_type, data_value),
              add0_weight_memref =
                  builder.create<mlir::bufferization::ToMemrefOp>(
                      builder.getUnknownLoc(), add0_weight_memref_type,
                      add0_weight_value),
              add1_weight_memref =
                  builder.create<mlir::bufferization::ToMemrefOp>(
                      builder.getUnknownLoc(), add1_weight_memref_type,
                      add1_weight_value);
  llvm::SmallVector<mlir::AffineExpr> input_exprs;
  for (size_t i = 0; i < input_rank; ++i) {
    input_exprs.push_back(mlir::getAffineDimExpr(i, context));
  }
  mlir::AffineMap indices_map = mlir::AffineMap::get(output_rank, 0,
                                                     input_exprs, context),
                  map = builder.getMultiDimIdentityMap(output_rank);
  llvm::SmallVector<mlir::AffineMap> maps = {indices_map, map, map, map};
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types;
  for (size_t i = 0; i < output_rank; ++i) {
    iterator_types.push_back(mlir::utils::IteratorType::parallel);
  }
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{input, add0_weight_memref, add1_weight_memref},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 4);
#endif
        mlir::Value first_index_value = b.create<mlir::arith::IndexCastOp>(
            loc, b.getIndexType(), inputs[0]);
        llvm::SmallVector<mlir::Value> indices_values;
        indices_values.push_back(first_index_value);
        for (size_t i = input_rank; i < output_rank; ++i) {
          mlir::Value value_index =
              b.create<mlir::linalg::IndexOp>(loc, b.getIndexType(), i);
          indices_values.push_back(std::move(value_index));
        }
        mlir::Value gather_op = b.create<mlir::memref::LoadOp>(loc, data_memref,
                                                               indices_values),
                    add0_op = b.create<mlir::arith::AddFOp>(loc, gather_op,
                                                            inputs[1]),
                    add1_op =
                        b.create<mlir::arith::AddFOp>(loc, add0_op, inputs[2]);
        b.create<mlir::linalg::YieldOp>(loc, add1_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers
