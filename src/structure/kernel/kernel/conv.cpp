#include "structure/kernel/kernel/conv.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "utils/type.h"
#include <cstdint>

namespace cpu_transformers {
namespace kernel {

ConvKernel::ConvKernel(std::vector<int64_t> &&dilations, int64_t group,
                       std::vector<int64_t> &&kernel_shape,
                       std::vector<int64_t> &&strides,
                       std::optional<Tensor> &&bias)
    : dilations_(std::move(dilations)), group_(group),
      kernel_shape_(std::move(kernel_shape)), strides_(std::move(strides)),
      bias_(std::move(bias)) {}

void ConvKernel::run(mlir::OpBuilder &builder, mlir::Value &input,
                     mlir::Value &weights, mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   weights_type =
                       mlir::cast<mlir::MemRefType>(weights.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  llvm::ArrayRef input_shape = input_type.getShape(),
                 weights_shape = weights_type.getShape(),
                 output_shape = output_type.getShape();
  const size_t rank = output_type.getRank();
#ifdef DEBUG
  assert(rank >= 2);
  assert(input_type.getRank() == rank);
  assert(weights_type.getRank() == rank);
  assert(input_shape.size() == rank);
  assert(weights_shape.size() == rank);
  assert(output_shape.size() == rank);
#endif
  const int64_t batch_size = input_shape[0], input_channels = input_shape[1],
                output_channels = output_shape[1];
#ifdef DEBUG
  assert(batch_size == output_shape[0]);
  assert(input_channels == weights_shape[1] || 1 == weights_shape[1]);
  assert(output_channels == weights_shape[0]);
  assert(input_channels % group_ == 0);
  assert(output_channels % group_ == 0);
#endif
  if (group_ == 1) {
    const size_t dims = rank - 2, loops = dims * 2 + 3;
    llvm::SmallVector<mlir::AffineExpr>
        input_exprs = {builder.getAffineDimExpr(0),
                       builder.getAffineDimExpr(2)},
        weights_exprs = {builder.getAffineDimExpr(1),
                         builder.getAffineDimExpr(2)},
        output_exprs = {builder.getAffineDimExpr(0),
                        builder.getAffineDimExpr(1)};
    llvm::SmallVector<mlir::utils::IteratorType> iterator_types(loops);
    iterator_types[0] = mlir::utils::IteratorType::parallel;
    iterator_types[1] = mlir::utils::IteratorType::parallel;
    iterator_types[2] = mlir::utils::IteratorType::reduction;
    for (size_t i = 0; i < dims; ++i) {
      mlir::AffineExpr index = builder.getAffineDimExpr(i + 3),
                       filter_size = builder.getAffineDimExpr(i + 3 + dims);
      input_exprs.push_back(index + filter_size);
      weights_exprs.push_back(filter_size);
      output_exprs.push_back(index);
      iterator_types[i + 3] = mlir::utils::IteratorType::parallel;
      iterator_types[i + 3 + dims] = mlir::utils::IteratorType::reduction;
    }
    mlir::AffineMap input_map =
                        mlir::AffineMap::get(loops, 0, input_exprs, context),
                    weights_map =
                        mlir::AffineMap::get(loops, 0, weights_exprs, context),
                    output_map =
                        mlir::AffineMap::get(loops, 0, output_exprs, context);
    mlir::Value c0f = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(),
        builder.getFloatAttr(output_type.getElementType(), 0.0));
    builder.create<mlir::linalg::FillOp>(builder.getUnknownLoc(), c0f, output);
    builder.create<mlir::linalg::GenericOp>(
        builder.getUnknownLoc(), mlir::TypeRange{},
        mlir::ValueRange{input, weights}, mlir::ValueRange{output},
        mlir::ArrayRef{input_map, weights_map, output_map}, iterator_types,
        [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
          assert(inputs.size() == 3);
#endif
          mlir::Value input = inputs[0], weights = inputs[1],
                      output = inputs[2],
                      mul_op =
                          b.create<mlir::arith::MulFOp>(loc, input, weights),
                      add_op =
                          b.create<mlir::arith::AddFOp>(loc, mul_op, output);
          b.create<mlir::linalg::YieldOp>(loc, add_op);
        });
  } else if (group_) {
    const size_t dims = rank - 2, loops = dims * 2 + 2;
    llvm::SmallVector<mlir::AffineExpr>
        input_exprs = {builder.getAffineDimExpr(0),
                       builder.getAffineDimExpr(1)},
        weights_exprs = {builder.getAffineDimExpr(1),
                         builder.getAffineConstantExpr(0)},
        output_exprs = {builder.getAffineDimExpr(0),
                        builder.getAffineDimExpr(1)};
    llvm::SmallVector<mlir::utils::IteratorType> iterator_types(loops);
    iterator_types[0] = mlir::utils::IteratorType::parallel;
    iterator_types[1] = mlir::utils::IteratorType::parallel;
    for (size_t i = 0; i < dims; ++i) {
      mlir::AffineExpr index = builder.getAffineDimExpr(i + 2),
                       filter_size = builder.getAffineDimExpr(i + 2 + dims);
      input_exprs.push_back(index + filter_size);
      weights_exprs.push_back(filter_size);
      output_exprs.push_back(index);
      iterator_types[i + 2] = mlir::utils::IteratorType::parallel;
      iterator_types[i + 2 + dims] = mlir::utils::IteratorType::reduction;
    }
    mlir::AffineMap input_map =
                        mlir::AffineMap::get(loops, 0, input_exprs, context),
                    weights_map =
                        mlir::AffineMap::get(loops, 0, weights_exprs, context),
                    output_map =
                        mlir::AffineMap::get(loops, 0, output_exprs, context);
    mlir::Value c0f = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(),
        builder.getFloatAttr(output_type.getElementType(), 0.0));
    builder.create<mlir::linalg::FillOp>(builder.getUnknownLoc(), c0f, output);
    builder.create<mlir::linalg::GenericOp>(
        builder.getUnknownLoc(), mlir::TypeRange{},
        mlir::ValueRange{input, weights}, mlir::ValueRange{output},
        mlir::ArrayRef{input_map, weights_map, output_map}, iterator_types,
        [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
          assert(inputs.size() == 3);
#endif
          mlir::Value input = inputs[0], weights = inputs[1],
                      output = inputs[2],
                      mul_op =
                          b.create<mlir::arith::MulFOp>(loc, input, weights),
                      add_op =
                          b.create<mlir::arith::AddFOp>(loc, mul_op, output);
          b.create<mlir::linalg::YieldOp>(loc, add_op);
        });
  } else {
#ifdef DEBUG
    assert(false && "unimplemented");
#else
    __builtin_unreachable();
#endif
  }
  if (bias_) {
    mlir::Value bias;
    const std::vector<float64_t> &bias_ref = bias_->Get();
    if (bias_->GetType() == Type::kFloat32) {
      llvm::SmallVector<float32_t> bias_data(bias_ref.begin(), bias_ref.end());
      const std::vector<int64_t> &bias_shape = bias_->GetShape();
      mlir::Type bias_type = builder.getF32Type();
      mlir::MemRefType bias_memref_type =
          mlir::MemRefType::get(bias_shape, bias_type);
      mlir::RankedTensorType bias_ranked_type =
          mlir::RankedTensorType::get(bias_shape, bias_type);
      mlir::DenseElementsAttr bias_attr = mlir::DenseElementsAttr::get(
          bias_ranked_type, llvm::ArrayRef(bias_data));
      mlir::Value bias_tensor = builder.create<mlir::arith::ConstantOp>(
          builder.getUnknownLoc(), bias_attr);
      bias = builder.create<mlir::bufferization::ToMemrefOp>(
          builder.getUnknownLoc(), bias_memref_type, bias_tensor);
    } else {
#ifdef DEBUG
      assert(false && "unimplemented");
#else
      __builtin_unreachable();
#endif
    }
    llvm::SmallVector<mlir::AffineExpr> input_exprs = {
        builder.getAffineConstantExpr(1)};
    mlir::AffineMap input_map =
        mlir::AffineMap::get(rank, 0, input_exprs, context);
    builder.create<mlir::linalg::GenericOp>(
        builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{bias},
        mlir::ValueRange{output},
        llvm::SmallVector<mlir::AffineMap>{
            input_map, builder.getMultiDimIdentityMap(rank)},
        llvm::SmallVector(rank, mlir::utils::IteratorType::parallel),
        [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
          assert(inputs.size() == 2);
#endif
          mlir::Value input = inputs[0], output = inputs[1],
                      add_op =
                          b.create<mlir::arith::AddFOp>(loc, input, output);
          b.create<mlir::linalg::YieldOp>(loc, add_op);
        });
  }
}

ConvWithoutPaddingKernel::ConvWithoutPaddingKernel(
    std::vector<int64_t> &&dilations, int64_t group,
    std::vector<int64_t> &&kernel_shape, std::vector<int64_t> &&strides,
    std::optional<Tensor> &&bias)
    : ConvKernel(std::move(dilations), group, std::move(kernel_shape),
                 std::move(strides), std::move(bias)) {}

std::string ConvWithoutPaddingKernel::GetKernelName() const {
  return kKernelName;
}

void ConvWithoutPaddingKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                                   mlir::Value &rhs,
                                   mlir::Value &output) const {
  run(builder, lhs, rhs, output);
}

ConvWithPaddingKernel::ConvWithPaddingKernel(
    std::vector<int64_t> &&dilations, int64_t group,
    std::vector<int64_t> &&kernel_shape, std::vector<int64_t> &&pads,
    std::vector<int64_t> &&strides, std::optional<Tensor> &&bias)
    : ConvKernel(std::move(dilations), group, std::move(kernel_shape),
                 std::move(strides), std::move(bias)),
      pads_(std::move(pads)) {}

std::string ConvWithPaddingKernel::GetKernelName() const { return kKernelName; }

void ConvWithPaddingKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                                mlir::Value &rhs, mlir::Value &output,
                                mlir::Value &buffer) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType lhs_type = mlir::cast<mlir::MemRefType>(lhs.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = lhs_type.getRank(), pad_rank = pads_.size();
  llvm::ArrayRef lhs_shape = lhs_type.getShape(),
                 output_shape = output_type.getShape();
  mlir::Type lhs_elem_type = lhs_type.getElementType(),
             output_elem_type = output_type.getElementType();
#ifdef DEBUG
  assert(pad_rank % 2 == 0);
  assert(pad_rank / 2 + 2 == rank);
#endif
  llvm::SmallVector<int64_t> offsets = {0, 0}, pad_shape(rank);
  pad_shape[0] = lhs_shape[0];
  pad_shape[1] = lhs_shape[1];
  for (size_t i = 0; i < pad_rank / 2; ++i) {
    int64_t offset = pads_[i];
    offsets.push_back(offset);
    pad_shape[i + 2] = lhs_shape[i + 2] + pads_[i] + pads_[i + pad_rank / 2];
  }
  mlir::MemRefType buf_memref_type =
      mlir::MemRefType::get(pad_shape, lhs_elem_type);
  mlir::Value c0 = builder.create<mlir::arith::ConstantOp>(
                  builder.getUnknownLoc(), builder.getIndexAttr(0)),
              c0f = builder.create<mlir::arith::ConstantOp>(
                  builder.getUnknownLoc(),
                  builder.getFloatAttr(lhs_elem_type, 0.0)),
              buf = builder.create<mlir::memref::ViewOp>(
                  builder.getUnknownLoc(), buf_memref_type, buffer, c0,
                  mlir::ValueRange{}),
              subview_op = builder.create<mlir::memref::SubViewOp>(
                  builder.getUnknownLoc(), buf, offsets, lhs_shape,
                  llvm::SmallVector<int64_t>(rank, 1));
  builder.create<mlir::linalg::FillOp>(builder.getUnknownLoc(), c0f, buf);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{lhs},
      mlir::ValueRange{subview_op},
      llvm::SmallVector(2, builder.getMultiDimIdentityMap(rank)),
      llvm::SmallVector(rank, mlir::utils::IteratorType::parallel),
      [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0];
        b.create<mlir::linalg::YieldOp>(loc, input);
      });
  run(builder, buf, rhs, output);
}

} // namespace kernel
} // namespace cpu_transformers