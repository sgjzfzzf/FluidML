#include "structure/kernel/matmul.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "structure/kernel/utils.h"
#include <cstdint>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

std::string MatMulKernel::GetKernelName() const { return kKernelName; }

void MatMulKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                       mlir::Value &rhs, mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType lhs_type = mlir::cast<mlir::MemRefType>(lhs.getType());
  mlir::MemRefType rhs_type = mlir::cast<mlir::MemRefType>(rhs.getType());
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  llvm::SmallVector<mlir::AffineMap> maps =
      GetBroadcastMatMulAffineMaps(context, lhs_type, rhs_type, output_type);
  const int64_t rank = output_type.getRank();
#ifdef DEBUG
  assert(rank >= 2);
#endif
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank + 1, mlir::utils::IteratorType::parallel);
  iterator_types[rank - 1] = mlir::utils::IteratorType::reduction;
  mlir::Value c0f = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getFloatAttr(output_type.getElementType(), 0.0));
  builder.create<mlir::linalg::FillOp>(builder.getUnknownLoc(), c0f, output);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value lhs = inputs[0], rhs = inputs[1], output = inputs[2],
                    mul_op = b.create<mlir::arith::MulFOp>(loc, lhs, rhs),
                    add_op = b.create<mlir::arith::AddFOp>(loc, mul_op, output);
        b.create<mlir::linalg::YieldOp>(loc, add_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers
