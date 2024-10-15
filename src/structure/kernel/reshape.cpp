#include "structure/kernel/reshape.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#ifdef DEBUG
#include <cassert>
#include <numeric>
#endif

namespace cpu_transformers {
namespace kernel {

class ReshapeKernelGeneratorImpl : public ReshapeKernelGenerator {
public:
  ReshapeKernelGeneratorImpl() = default;
  ReshapeKernelGeneratorImpl(const ReshapeKernelGeneratorImpl &generator) =
      delete;
  ReshapeKernelGeneratorImpl(ReshapeKernelGeneratorImpl &&generator) = default;
  virtual ~ReshapeKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<ReshapeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
};

std::string ReshapeKernel::GetKernelName() const { return kKernelName; }

void ReshapeKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                        mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  llvm::ArrayRef<int64_t> input_shape = input_type.getShape(),
                          output_shape = output_type.getShape();
  const int64_t elem_num = input_type.getNumElements(),
                input_rank = input_shape.size(),
                output_rank = output_shape.size();
#ifdef DEBUG
  assert(elem_num == std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                     std::multiplies<int64_t>()));
  assert(elem_num == output_type.getNumElements());
  assert(elem_num == std::accumulate(output_shape.begin(), output_shape.end(),
                                     1, std::multiplies<int64_t>()));
#endif
  builder.create<mlir::affine::AffineForOp>(
      builder.getUnknownLoc(), 0, elem_num, 1, std::nullopt,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value arg,
          mlir::ValueRange args) {
        llvm::SmallVector<mlir::Value> input_indices, output_indices;
        for (size_t i = 0, prod = elem_num; i < input_rank; ++i) {
          mlir::AffineExpr expr = builder.getAffineDimExpr(0) % prod;
          prod /= input_shape[i];
          expr = expr.floorDiv(prod);
          mlir::AffineMap map = mlir::AffineMap::get(1, 0, expr, context);
          mlir::Value index =
              b.create<mlir::affine::AffineApplyOp>(loc, map, arg);
          input_indices.push_back(std::move(index));
        }
        for (size_t i = 0, prod = elem_num; i < output_rank; ++i) {
          mlir::AffineExpr expr = builder.getAffineDimExpr(0) % prod;
          prod /= output_shape[i];
          expr = expr.floorDiv(prod);
          mlir::AffineMap map = mlir::AffineMap::get(1, 0, expr, context);
          mlir::Value index =
              b.create<mlir::affine::AffineApplyOp>(loc, map, arg);
          output_indices.push_back(std::move(index));
        }
        mlir::Value value =
            b.create<mlir::affine::AffineLoadOp>(loc, input, input_indices);
        b.create<mlir::affine::AffineStoreOp>(loc, std::move(value), output,
                                              output_indices);
        b.create<mlir::affine::AffineYieldOp>(loc);
      });
}

std::unique_ptr<ReshapeKernelGenerator> ReshapeKernelGenerator::Make() {
  return std::make_unique<ReshapeKernelGeneratorImpl>();
}

std::shared_ptr<SingleInputWithoutBufferKernel>
ReshapeKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<ReshapeKernel>
ReshapeKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                  llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<ReshapeKernel>();
}

} // namespace kernel
} // namespace cpu_transformers
