#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_UTILS_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_UTILS_H_

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace cpu_transformers {
namespace kernel {

enum Axis : size_t { i, j, k };

llvm::SmallVector<llvm::SmallVector<Axis, 3>> GetAxesInAllOrders();

llvm::SmallVector<mlir::AffineMap>
GetBroadcastAffineMaps(mlir::Builder &builder,
                       llvm::ArrayRef<mlir::MemRefType> input_types,
                       const mlir::MemRefType &output_type);

llvm::SmallVector<mlir::AffineMap> GetBroadcastMatMulAffineMaps(
    mlir::MLIRContext *context, const mlir::MemRefType &lhs_type,
    const mlir::MemRefType &rhs_type, const mlir::MemRefType &output_type);

llvm::SmallVector<mlir::AffineMap> GetBroadcastMatMulAffineMaps(
    mlir::MLIRContext *context, const mlir::MemRefType &lhs_type,
    const mlir::MemRefType &rhs_type, const mlir::MemRefType &output_type,
    llvm::ArrayRef<Axis> axes);

} // namespace kernel
} // namespace cpu_transformers

#endif
