#include "structure/kernel/kernel.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#ifdef DEBUG
#include "exception/unreachable_exception.h"
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {
llvm::SmallVector<mlir::AffineMap>
Kernel::getBroadcastAffineMaps(mlir::Builder &builder,
                               llvm::ArrayRef<mlir::MemRefType> input_types,
                               const mlir::MemRefType &output_type) {
  llvm::ArrayRef<int64_t> output_shape = output_type.getShape();
  const size_t rank = output_shape.size();
  const size_t size = input_types.size();
  llvm::SmallVector<llvm::SmallVector<mlir::AffineExpr>> exprs(size);
  for (size_t i = 0; i < rank; ++i) {
    for (size_t j = 0; j < size; ++j) {
      const mlir::MemRefType &input_type = input_types[j];
      const llvm::ArrayRef<int64_t> input_shape = input_type.getShape();
      const size_t len = input_shape.size();
      const int64_t index = len - rank + i;
      if (index >= 0) {
        const int64_t dim = input_shape[index];
        if (dim == output_shape[i]) {
          exprs[j].push_back(builder.getAffineDimExpr(i));
        } else if (dim == 1) {
          exprs[j].push_back(builder.getAffineConstantExpr(0));
        } else {
#ifdef DEBUG
          throw UnreachableException();
#else
          __builtin_unreachable();
#endif
        }
      }
    }
  }
  llvm::SmallVector<mlir::AffineMap> maps;
  for (size_t i = 0; i < size; ++i) {
    maps.push_back(
        mlir::AffineMap::get(rank, 0, exprs[i], builder.getContext()));
  }
  maps.push_back(builder.getMultiDimIdentityMap(rank));
  return maps;
}

llvm::SmallVector<mlir::AffineMap> Kernel::getBroadcastMatMulAffineMaps(
    mlir::MLIRContext *context, const mlir::MemRefType &lhs_type,
    const mlir::MemRefType &rhs_type, const mlir::MemRefType &output_type) {
  llvm::ArrayRef<int64_t> lhs_shape = lhs_type.getShape(),
                          rhs_shape = rhs_type.getShape(),
                          output_shape = output_type.getShape();
  const size_t lhs_rank = lhs_shape.size(), rhs_rank = rhs_shape.size(),
               output_rank = output_shape.size();
#ifdef DEBUG
  assert(lhs_rank >= 2);
  assert(rhs_rank >= 2);
  assert(output_rank >= 2);
  assert(lhs_rank <= output_rank);
  assert(rhs_rank <= output_rank);
#endif
  llvm::SmallVector<mlir::AffineExpr> lhs_exprs, rhs_exprs, output_exprs;
  for (size_t i = 0; i < output_rank - 2; ++i) {
    const int64_t lhs_index = lhs_rank - output_rank + i,
                  rhs_index = rhs_rank - output_rank + i, output_index = i;
    const int64_t output_dim = output_shape[output_index];
    if (lhs_index >= 0) {
      const int64_t lhs_dim = lhs_shape[lhs_index];
      if (lhs_dim == output_dim) {
        lhs_exprs.push_back(mlir::getAffineDimExpr(lhs_index, context));
      } else if (lhs_dim == 1) {
        lhs_exprs.push_back(mlir::getAffineConstantExpr(0, context));
      } else {
#ifdef DEBUG
        throw UnreachableException();
#else
        __builtin_unreachable();
#endif
      }
    }
    if (rhs_index >= 0) {
      const int64_t rhs_dim = rhs_shape[rhs_index];
      if (rhs_dim == output_dim) {
        rhs_exprs.push_back(mlir::getAffineDimExpr(rhs_index, context));
      } else if (rhs_dim == 1) {
        rhs_exprs.push_back(mlir::getAffineConstantExpr(0, context));
      } else {
#ifdef DEBUG
        throw UnreachableException();
#else
        __builtin_unreachable();
#endif
      }
    }
    output_exprs.push_back(mlir::getAffineDimExpr(output_index, context));
  }
  // the order of axes is m, k, n
  lhs_exprs.push_back(mlir::getAffineDimExpr(output_rank - 2, context));
  lhs_exprs.push_back(mlir::getAffineDimExpr(output_rank - 1, context));
  rhs_exprs.push_back(mlir::getAffineDimExpr(output_rank - 1, context));
  rhs_exprs.push_back(mlir::getAffineDimExpr(output_rank, context));
  output_exprs.push_back(mlir::getAffineDimExpr(output_rank - 2, context));
  output_exprs.push_back(mlir::getAffineDimExpr(output_rank, context));
  mlir::AffineMap lhs_map = mlir::AffineMap::get(output_rank + 1, 0, lhs_exprs,
                                                 context),
                  rhs_map = mlir::AffineMap::get(output_rank + 1, 0, rhs_exprs,
                                                 context),
                  output_map = mlir::AffineMap::get(output_rank + 1, 0,
                                                    output_exprs, context);
  return {lhs_map, rhs_map, output_map};
}
} // namespace kernel
} // namespace cpu_transformers
