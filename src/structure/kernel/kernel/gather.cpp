#include "structure/kernel/kernel/gather.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "utils/float.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#ifdef DEBUG
#include <cassert>
#endif

namespace {
void createNestLoops(mlir::OpBuilder &builder, mlir::Location loc, size_t i,
                     size_t axis, llvm::ArrayRef<int64_t> input_shape,
                     llvm::ArrayRef<int64_t> output_shape, mlir::Value &input,
                     mlir::Value &indices, mlir::Value &output,
                     llvm::SmallVector<mlir::Value> &&output_indices) {
  const size_t rank = output_shape.size();
  mlir::Value c0 = builder.create<mlir::arith::ConstantOp>(
                  loc, builder.getIndexAttr(0)),
              c1 = builder.create<mlir::arith::ConstantOp>(
                  loc, builder.getIndexAttr(1)),
              end = builder.create<mlir::arith::ConstantOp>(
                  loc, builder.getIndexAttr(output_shape[i]));
  if (i == rank - 1) {
    builder.create<mlir::scf::ForOp>(
        loc, c0, end, c1, std::nullopt,
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value arg,
            mlir::ValueRange args) {
          output_indices.push_back(arg);
          llvm::SmallVector<mlir::Value> index_indices;
          const size_t input_rank = input_shape.size(),
                       output_rank = output_shape.size(),
                       indices_rank = output_rank - input_rank + 1;
#ifdef DEBUG
          assert(output_indices.size() == output_rank);
#endif
          for (size_t i = axis; i < axis + indices_rank; ++i) {
            index_indices.push_back(output_indices[i]);
          }
#ifdef DEBUG
          assert(index_indices.size() == indices_rank);
#endif
          mlir::Value index =
              b.create<mlir::memref::LoadOp>(loc, indices, index_indices);
          index =
              b.create<mlir::arith::IndexCastOp>(loc, b.getIndexType(), index);
          llvm::SmallVector<mlir::Value> input_indices;
          for (size_t i = 0; i < axis; ++i) {
            input_indices.push_back(output_indices[i]);
          }
          input_indices.push_back(index);
          for (size_t i = axis + indices_rank; i < output_rank; ++i) {
            input_indices.push_back(output_indices[i]);
          }
#ifdef DEBUG
          assert(input_indices.size() == input_rank);
#endif
          mlir::Value value =
              b.create<mlir::memref::LoadOp>(loc, input, input_indices);
          b.create<mlir::memref::StoreOp>(loc, value, output, output_indices);
          b.create<mlir::scf::YieldOp>(loc);
        });
  } else {
    builder.create<mlir::scf::ForOp>(
        loc, c0, end, c1, std::nullopt,
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value arg,
            mlir::ValueRange args) {
          output_indices.push_back(arg);
          createNestLoops(b, loc, i + 1, axis, input_shape, output_shape, input,
                          indices, output, std::move(output_indices));
          builder.create<mlir::scf::YieldOp>(loc);
        });
  }
}
} // namespace

namespace fluidml {
namespace kernel {

std::string GatherConstantIndexScalarKernel::GetKernelName() const {
  return kKernelName;
}

GatherConstantIndexScalarKernel::GatherConstantIndexScalarKernel(int64_t axis,
                                                                 int64_t index)
    : axis_(axis), index_(index) {}

void GatherConstantIndexScalarKernel::Run(mlir::OpBuilder &builder,
                                          mlir::Value &input,
                                          mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType data_memref_type =
                       mlir::cast<mlir::MemRefType>(input.getType()),
                   output_memref_type =
                       mlir::cast<mlir::MemRefType>(output.getType());
  const int64_t data_rank = data_memref_type.getRank(),
                output_rank = output_memref_type.getRank();
#ifdef DEBUG
  assert(data_rank - 1 == output_rank);
#endif
  const size_t axis = axis_ >= 0 ? axis_ : data_rank + axis_;
  llvm::SmallVector<mlir::AffineExpr> data_exprs;
  for (size_t i = 0; i < data_rank; ++i) {
    if (i == axis_) {
#ifdef DEBUG
      assert(index_ >= 0);
#endif
      data_exprs.push_back(mlir::getAffineConstantExpr(index_, context));
    } else {
      const size_t index = i < axis_ ? i : i - 1;
      data_exprs.push_back(mlir::getAffineDimExpr(index, context));
    }
  }
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output},
      llvm::ArrayRef{mlir::AffineMap::get(output_rank, 0, data_exprs, context),
                     builder.getMultiDimIdentityMap(output_rank)},
      llvm::SmallVector(output_rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value lhs = inputs[0];
        b.create<mlir::linalg::YieldOp>(loc, lhs);
      });
}

GatherConstantIndicesTensorKernel::GatherConstantIndicesTensorKernel(
    Tensor &&indices, int64_t axis)
    : indices_(std::move(indices)), axis_(axis) {}

std::string GatherConstantIndicesTensorKernel::GetKernelName() const {
  return kKernelName;
}

void GatherConstantIndicesTensorKernel::Run(mlir::OpBuilder &builder,
                                            mlir::Value &input,
                                            mlir::Value &output) const {
#ifdef DEBUG
  assert(axis_ >= 0);
#endif
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  llvm::ArrayRef input_shape = input_type.getShape(),
                 output_shape = output_type.getShape();
  const size_t input_rank = input_shape.size(),
               output_rank = output_shape.size();
  mlir::DenseElementsAttr elements;
  mlir::MemRefType indices_memref_type;
  if (indices_.GetType() == Type::kInt64) {
    const std::vector<float64_t> indices_ref = indices_.Get();
    llvm::SmallVector<int64_t> indices(indices_ref.begin(), indices_ref.end());
    indices_memref_type =
        mlir::MemRefType::get(indices_.GetShape(), builder.getI64Type());
    elements = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get(indices_.GetShape(), builder.getI64Type()),
        llvm::ArrayRef(indices));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  mlir::Value indices = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), elements);
  indices = builder.create<mlir::bufferization::ToMemrefOp>(
      builder.getUnknownLoc(), indices_memref_type, indices);
  createNestLoops(builder, builder.getUnknownLoc(), 0, axis_, input_shape,
                  output_shape, input, indices, output,
                  llvm::SmallVector<mlir::Value>{});
}

GatherConstantDataTensorKernel::GatherConstantDataTensorKernel(Tensor &&data)
    : data_(std::move(data)) {}

std::string GatherConstantDataTensorKernel::GetKernelName() const {
  return kKernelName;
}

void GatherConstantDataTensorKernel::Run(mlir::OpBuilder &builder,
                                         mlir::Value &input,
                                         mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType indices_memref_type =
      mlir::cast<mlir::MemRefType>(input.getType());
  mlir::MemRefType output_memref_type =
      mlir::cast<mlir::MemRefType>(output.getType());
  const std::vector<int64_t> &data_shape = data_.GetShape();
  const std::vector<float64_t> &data_ref = data_.Get();
  const int64_t data_rank = data_shape.size(),
                indices_rank = indices_memref_type.getRank(),
                output_rank = output_memref_type.getRank();
  llvm::ArrayRef<int64_t> indices_shape = indices_memref_type.getShape(),
                          output_shape = output_memref_type.getShape();
#ifdef DEBUG
  assert(data_rank + indices_rank - 1 == output_rank);
  for (size_t i = 0; i < output_rank; ++i) {
    if (i < indices_rank) {
      assert(indices_shape[i] == output_shape[i]);
    } else {
      assert(data_shape[i - indices_rank + 1] == output_shape[i]);
    }
  }
#endif
  mlir::ShapedType data_shaped_type =
      mlir::RankedTensorType::get(data_shape, mlir::FloatType::getF32(context));
  mlir::DenseElementsAttr elements;
  if (data_.GetType() == Type::kFloat32) {
    llvm::SmallVector<float32_t> data(data_ref.begin(), data_ref.end());
    elements =
        mlir::DenseElementsAttr::get(data_shaped_type, llvm::ArrayRef(data));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  mlir::Value data_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), elements);
  mlir::MemRefType data_memref_type = mlir::MemRefType::get(
      data_shape, mlir::FloatType::getF32(context), {}, 0);
  mlir::Value data_memref_ref = builder.create<mlir::bufferization::ToMemrefOp>(
      builder.getUnknownLoc(), data_memref_type, data_value);
  llvm::SmallVector<mlir::AffineExpr> input_exprs;
  for (size_t i = 0; i < indices_rank; ++i) {
    input_exprs.push_back(mlir::getAffineDimExpr(i, context));
  }
  mlir::AffineMap indices_map =
      mlir::AffineMap::get(output_rank, 0, input_exprs, context);
  mlir::AffineMap output_map = builder.getMultiDimIdentityMap(output_rank);
  llvm::SmallVector<mlir::AffineMap> maps = {indices_map, output_map};
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types;
  for (size_t i = 0; i < output_rank; ++i) {
    iterator_types.push_back(mlir::utils::IteratorType::parallel);
  }
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value first_index_value = b.create<mlir::arith::IndexCastOp>(
            loc, b.getIndexType(), inputs[0]);
        llvm::SmallVector<mlir::Value> indices_values;
        indices_values.push_back(first_index_value);
        for (size_t i = indices_rank; i < output_rank; ++i) {
          mlir::Value value_index =
              b.create<mlir::linalg::IndexOp>(loc, b.getIndexType(), i);
          indices_values.push_back(std::move(value_index));
        }
        mlir::Value out = b.create<mlir::memref::LoadOp>(loc, data_memref_ref,
                                                         indices_values);
        b.create<mlir::linalg::YieldOp>(loc, out);
      });
}

} // namespace kernel
} // namespace fluidml
