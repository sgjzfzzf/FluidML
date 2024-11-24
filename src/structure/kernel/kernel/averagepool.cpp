#include "structure/kernel/kernel/averagepool.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "utils/utils.h"
#include <numeric>

namespace fluidml {
namespace kernel {

AveragePoolKernel::AveragePoolKernel(std::vector<int64_t> &&dilations,
                                     std::vector<int64_t> &&kernel_shape,
                                     std::vector<int64_t> &&strides)
    : dilations_(std::move(dilations)), kernel_shape_(std::move(kernel_shape)),
      strides_(std::move(strides)) {}

void AveragePoolKernel::run(mlir::OpBuilder &builder, mlir::Value &input,
                            mlir::Value &output) const {
  const mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = input.getType().cast<mlir::MemRefType>(),
                   output_type = output.getType().cast<mlir::MemRefType>();
  llvm::ArrayRef input_shape = input_type.getShape(),
                 output_shape = output_type.getShape();
  const size_t rank = input_type.getRank(), kernel_rank = kernel_shape_.size(),
               batches = rank - kernel_rank,
               kernel_size =
                   std::accumulate(kernel_shape_.begin(), kernel_shape_.end(),
                                   1, std::multiplies<int64_t>());
#ifdef DEBUG
  assert(rank == output_type.getRank());
#endif
  std::vector<std::vector<size_t>> indices =
      utils::GenAllIndicesInOrder(kernel_shape_);
  llvm::SmallVector<mlir::Value> subviews;
  for (const std::vector<size_t> &index : indices) {
    llvm::SmallVector<int64_t> offsets(rank, 0), sizes(rank, 1),
        strides(rank, 1);
    for (size_t i = 0; i < batches; ++i) {
#ifdef DEBUG
      assert(input_shape[i] == output_shape[i]);
#endif
      sizes[i] = input_shape[i];
    }
    for (size_t i = 0; i < kernel_rank; ++i) {
      const int64_t output_dim = output_shape[i + batches],
                    kernel_dim = kernel_shape_[i], offset = index[i],
                    stride = strides_[i];
      offsets[i + batches] = offset;
      sizes[i + batches] = output_dim;
      strides[i + batches] = stride;
    }
    mlir::Value subview = builder.create<mlir::memref::SubViewOp>(
        builder.getUnknownLoc(), input, offsets, sizes, strides);
    subviews.push_back(std::move(subview));
  }
  llvm::SmallVector<mlir::AffineMap> maps(subviews.size() + 1,
                                          builder.getMultiDimIdentityMap(rank));
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, subviews,
      mlir::ValueRange{output},
      llvm::SmallVector(subviews.size() + 1,
                        builder.getMultiDimIdentityMap(rank)),
      llvm::SmallVector<mlir::utils::IteratorType>(
          rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange values) {
#ifdef DEBUG
        assert(values.size() == subviews.size() + 1);
#endif
        mlir::Value sum = values[0], output_op;
        mlir::Type type = sum.getType();
        if (type.isa<mlir::FloatType>()) {
          mlir::Value factor = builder.create<mlir::arith::ConstantOp>(
              loc, builder.getFloatAttr(type, kernel_size));
          for (size_t i = 1; i < values.size(); ++i) {
            sum = builder.create<mlir::arith::AddFOp>(loc, sum, values[i]);
          }
          output_op = builder.create<mlir::arith::DivFOp>(loc, sum, factor);
        } else {
#ifdef DEBUG
          assert(false && "unimplemented");
#endif
        }
        b.create<mlir::linalg::YieldOp>(loc, output_op);
      });
}

AveragePoolWithoutPaddingKernel::AveragePoolWithoutPaddingKernel(
    std::vector<int64_t> &&dilations, std::vector<int64_t> &&kernel_shape,
    std::vector<int64_t> &&strides)
    : AveragePoolKernel(std::move(dilations), std::move(kernel_shape),
                        std::move(strides)) {}

std::string AveragePoolWithoutPaddingKernel::GetKernelName() const {
  return kKernelName;
}

void AveragePoolWithoutPaddingKernel::Run(mlir::OpBuilder &builder,
                                          mlir::Value &input,
                                          mlir::Value &output) const {
  run(builder, input, output);
}

} // namespace kernel
} // namespace fluidml
