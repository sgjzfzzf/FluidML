#include "structure/kernel/kernel/cum_sum.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

namespace {
using LastLoopBodyFn =
    std::function<void(mlir::OpBuilder &, mlir::Location, mlir::ValueRange,
                       llvm::SmallVector<mlir::Value> &&)>;

void createNestLoops(mlir::OpBuilder &builder, mlir::Location loc, size_t i,
                     llvm::ArrayRef<int64_t> shape, LastLoopBodyFn &&fn,
                     mlir::Value &init,
                     llvm::SmallVector<mlir::Value> &&indices) {
  const size_t rank = shape.size();
  if (i == rank - 1) {
    builder.create<mlir::affine::AffineForOp>(
        loc, 0, shape[i], 1, init,
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value arg,
            mlir::ValueRange args) {
          indices.push_back(arg);
          fn(b, loc, args, std::move(indices));
        });
  } else {
    builder.create<mlir::affine::AffineForOp>(
        loc, 0, shape[i], 1, std::nullopt,
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value arg,
            mlir::ValueRange args) {
          indices.push_back(arg);
          createNestLoops(b, loc, i + 1, shape, std::move(fn), init,
                          std::move(indices));
          builder.create<mlir::affine::AffineYieldOp>(loc);
        });
  }
}

} // namespace

namespace fluidml {
namespace kernel {

CumSumKernel::CumSumKernel(int64_t axis, bool exclusive, bool reverse)
    : axis_(axis), exclusive_(exclusive), reverse_(reverse) {
// TODO: we don't support exclusive and reverse yet
#ifdef DEBUG
  assert(!exclusive);
  assert(!reverse);
#endif
}

std::string CumSumKernel::GetKernelName() const { return kKernelName; }

void CumSumKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                       mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  llvm::ArrayRef<int64_t> shape = input_type.getShape();
  const size_t rank = shape.size();
  mlir::Type elem_type = input_type.getElementType();
#ifdef DEBUG
  assert(axis_ >= 0 && axis_ < rank);
  assert(shape == output_type.getShape());
  assert(rank == output_type.getRank());
  assert(elem_type == output_type.getElementType());
#endif
  mlir::Value c0 = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getIntegerAttr(elem_type, 0));
  llvm::SmallVector<int64_t> axes(shape.begin(), shape.end());
  std::swap(axes[axis_], axes[rank - 1]);
  createNestLoops(
      builder, builder.getUnknownLoc(), 0, axes,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args,
          llvm::SmallVector<mlir::Value> &&indices) {
#ifdef DEBUG
        assert(args.size() == 1);
#endif
        std::swap(indices[axis_], indices[rank - 1]);
        mlir::Value arg = args[0],
                    load_op = b.create<mlir::affine::AffineLoadOp>(loc, input,
                                                                   indices),
                    add_op = b.create<mlir::arith::AddIOp>(loc, load_op, arg);
        b.create<mlir::affine::AffineStoreOp>(loc, add_op, output, indices);
        b.create<mlir::affine::AffineYieldOp>(loc, add_op);
      },
      c0, llvm::SmallVector<mlir::Value>{});
}

} // namespace kernel
} // namespace fluidml
