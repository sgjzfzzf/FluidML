#ifndef FLUIDML_STRUCTURE_KERNEL_KERNEL_UTILS_H_
#define FLUIDML_STRUCTURE_KERNEL_KERNEL_UTILS_H_

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace fluidml {
namespace kernel {

enum Axis : size_t { i, j, k };

llvm::SmallVector<llvm::SmallVector<Axis, 3>> GetAxesInAllOrders();

llvm::SmallVector<mlir::AffineMap>
GetBroadcastAffineMaps(mlir::Builder &builder,
                       llvm::ArrayRef<mlir::MemRefType> input_types,
                       const mlir::MemRefType &output_type);

std::tuple<llvm::SmallVector<mlir::AffineMap>,
           llvm::SmallVector<mlir::utils::IteratorType>>
GetBroadcastMatMul(mlir::MLIRContext *context, const mlir::MemRefType &lhs_type,
                   const mlir::MemRefType &rhs_type,
                   const mlir::MemRefType &output_type);

std::tuple<llvm::SmallVector<mlir::AffineMap>,
           llvm::SmallVector<mlir::utils::IteratorType>>
GetBroadcastMatMul(mlir::MLIRContext *context, const mlir::MemRefType &lhs_type,
                   const mlir::MemRefType &rhs_type,
                   const mlir::MemRefType &output_type,
                   llvm::ArrayRef<Axis> axes);

} // namespace kernel
} // namespace fluidml

#endif
