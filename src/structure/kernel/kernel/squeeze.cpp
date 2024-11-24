#include "structure/kernel/kernel/squeeze.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#ifdef DEBUG
#include <cassert>
#endif

namespace fluidml {
namespace kernel {

SqueezeKernel::SqueezeKernel(std::vector<int64_t> &&axes)
    : axes_(std::move(axes)) {}

std::string SqueezeKernel::GetKernelName() const { return kKernelName; }

void SqueezeKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                        mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t input_rank = input_type.getRank(),
               output_rank = output_type.getRank();
#ifdef DEBUG
  assert(input_rank - axes_.size() == output_rank);
#endif
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (size_t i = 0; i < input_rank; ++i) {
    if (std::find(axes_.begin(), axes_.end(), i) == axes_.end()) {
      exprs.push_back(builder.getAffineDimExpr(i));
    } else {
      exprs.push_back(builder.getAffineConstantExpr(0));
    }
  }
#ifdef DEBUG
  assert(exprs.size() == input_rank);
#endif
  mlir::AffineMap input_map =
                      mlir::AffineMap::get(output_rank, 0, exprs, context),
                  output_map = builder.getMultiDimIdentityMap(output_rank);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output},
      llvm::ArrayRef<mlir::AffineMap>{input_map, output_map},
      llvm::SmallVector<mlir::utils::IteratorType>(
          output_rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0];
        b.create<mlir::linalg::YieldOp>(loc, input);
      });
}

} // namespace kernel
} // namespace fluidml
