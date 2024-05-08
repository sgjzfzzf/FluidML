#ifndef CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_H_
#define CPU_TRANSFORMERS_STRUCTURE_KERNEL_KERNEL_H_

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace cpu_transformers {
namespace kernel {
class Kernel {
public:
  Kernel() = default;
  Kernel(const Kernel &kernel) = delete;
  Kernel(Kernel &&kernel) = default;
  virtual ~Kernel() = default;

protected:
  static llvm::SmallVector<mlir::AffineMap>
  getBroadcastAffineMaps(mlir::Builder &builder,
                         llvm::ArrayRef<mlir::MemRefType> input_types,
                         const mlir::MemRefType &output_type);
  static llvm::SmallVector<mlir::AffineMap> getBroadcastMatMulAffineMaps(
      mlir::MLIRContext *context, const mlir::MemRefType &lhs_type,
      const mlir::MemRefType &rhs_type, const mlir::MemRefType &output_type);
};
} // namespace kernel
} // namespace cpu_transformers

#endif
