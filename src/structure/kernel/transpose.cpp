#include "structure/kernel/transpose.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

TransposeKernel::TransposeKernel(std::vector<int64_t> perms) : perms_(perms) {}

void TransposeKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                          mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  int64_t rank = perms_.size();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
#ifdef DEBUG
  assert(rank == input_type.getRank());
  assert(rank == output_type.getRank());
#endif
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (int64_t perm : perms_) {
    exprs.push_back(builder.getAffineDimExpr(perm));
  }
  mlir::AffineMap input_map = builder.getMultiDimIdentityMap(rank),
                  output_map = mlir::AffineMap::get(rank, 0, exprs, context);
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output},
      llvm::ArrayRef<mlir::AffineMap>{input_map, output_map}, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0];
        b.create<mlir::linalg::YieldOp>(loc, input);
      });
}

} // namespace kernel
} // namespace cpu_transformers
