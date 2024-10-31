#include "structure/kernel/kernel/maxpool.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "utils/utils.h"
#include <numeric>

namespace fluidml {
namespace kernel {

MaxPoolKernel::MaxPoolKernel(std::vector<int64_t> &&kernel_shape,
                             std::vector<int64_t> &&strides)
    : kernel_shape_(std::move(kernel_shape)), strides_(std::move(strides)) {}

void MaxPoolKernel::run(mlir::OpBuilder &builder, mlir::Value &input,
                        mlir::Value &output) const {
  mlir::MemRefType input_type = input.getType().cast<mlir::MemRefType>(),
                   output_type = output.getType().cast<mlir::MemRefType>();
  llvm::ArrayRef input_shape = input_type.getShape();
  const size_t rank = input_type.getRank(), kernel_rank = kernel_shape_.size();
#ifdef DEBUG
  llvm::ArrayRef output_shape = output_type.getShape();
  assert(rank == output_type.getRank());
  assert(rank - 2 == kernel_rank);
  assert(input_shape[0] == output_shape[0]);
  assert(input_shape[1] == output_shape[1]);
#endif
  std::vector<std::vector<size_t>> indices =
      utils::GenAllIndicesInOrder(kernel_shape_);
#ifdef DEBUG
  assert(indices.size() == std::accumulate(kernel_shape_.begin(),
                                           kernel_shape_.end(), 1,
                                           std::multiplies<int64_t>()));
#endif
  llvm::SmallVector<mlir::Value> subviews;
  for (const std::vector<size_t> &index : indices) {
#ifdef DEBUG
    assert(index.size() == kernel_rank);
#endif
    llvm::SmallVector<int64_t> offsets(rank, 0), sizes(rank, 1),
        strides(rank, 1);
    sizes[0] = input_shape[0];
    sizes[1] = input_shape[1];
    for (size_t i = 0; i < kernel_rank; ++i) {
      const int64_t dim = input_shape[i + 2], offset = index[i],
                    stride = strides_[i];
      offsets[i + 2] = offset;
      sizes[i + 2] = (dim - offset + 1) / stride;
      strides[i + 2] = stride;
    }
    mlir::Value subview = builder.create<mlir::memref::SubViewOp>(
        builder.getUnknownLoc(), input, offsets, sizes, strides);
    subviews.push_back(subview);
  }
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, subviews,
      mlir::ValueRange{output},
      llvm::SmallVector(subviews.size() + 1,
                        builder.getMultiDimIdentityMap(rank)),
      llvm::SmallVector<mlir::utils::IteratorType>(
          rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == subviews.size() + 1);
#endif
        mlir::Value max = inputs[0], output = inputs.back();
        mlir::Type type = output.getType();
#ifdef DEBUG
        assert(type == max.getType());
        assert(type.isa<mlir::FloatType>());
#endif
        const size_t upper = inputs.size() - 1;
        for (size_t i = 1; i < upper; ++i) {
          max = b.create<mlir::arith::MaximumFOp>(loc, max, inputs[i]);
        }
        b.create<mlir::linalg::YieldOp>(loc, max);
      });
}

MaxPoolWithoutPaddingKernel::MaxPoolWithoutPaddingKernel(
    std::vector<int64_t> &&kernel_shape, std::vector<int64_t> &&strides)
    : MaxPoolKernel(std::move(kernel_shape), std::move(strides)) {}

std::string MaxPoolWithoutPaddingKernel::GetKernelName() const {
  return kKernelName;
}

void MaxPoolWithoutPaddingKernel::Run(mlir::OpBuilder &builder,
                                      mlir::Value &input,
                                      mlir::Value &output) const {
  run(builder, input, output);
}

} // namespace kernel
} // namespace fluidml
