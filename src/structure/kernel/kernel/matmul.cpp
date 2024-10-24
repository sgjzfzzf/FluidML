#include "structure/kernel/kernel/matmul.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "structure/kernel/kernel/utils.h"
#include <cstdint>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

MatMulKernel::MatMulKernel() : axes_({Axis::i, Axis::j, Axis::k}) {}

MatMulKernel::MatMulKernel(llvm::SmallVector<Axis, 3> &&axes)
    : axes_(std::move(axes)) {}

std::string MatMulKernel::GetKernelName() const { return kKernelName; }

void MatMulKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                       mlir::Value &rhs, mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType lhs_type = mlir::cast<mlir::MemRefType>(lhs.getType()),
                   rhs_type = mlir::cast<mlir::MemRefType>(rhs.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  auto [maps, iterator_types] =
      GetBroadcastMatMul(context, lhs_type, rhs_type, output_type, axes_);
  const int64_t rank = output_type.getRank();
#ifdef DEBUG
  assert(rank >= 2);
#endif
  size_t k_index =
      std::find(axes_.begin(), axes_.end(), Axis::k) - axes_.begin();
  std::vector<int64_t> a(axes_.begin(), axes_.end());
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
