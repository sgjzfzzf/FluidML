#include "structure/kernel/unsqueeze.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

class UnSqueezeKernelGeneratorImpl : public UnSqueezeKernelGenerator {
public:
  UnSqueezeKernelGeneratorImpl(std::vector<int64_t> axes);
  UnSqueezeKernelGeneratorImpl(const UnSqueezeKernelGeneratorImpl &generator) =
      delete;
  UnSqueezeKernelGeneratorImpl(UnSqueezeKernelGeneratorImpl &&generator) =
      default;
  virtual ~UnSqueezeKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<UnSqueezeKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const std::vector<int64_t> axes_;
};

UnSqueezeKernel::UnSqueezeKernel(std::vector<int64_t> &&axes)
    : axes_(std::move(axes)) {}

std::string UnSqueezeKernel::GetKernelName() const { return kKernelName; }

void UnSqueezeKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                          mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t output_rank = output_type.getRank();
#ifdef DEBUG
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType());
  const size_t input_rank = input_type.getRank();
  assert(input_rank + axes_.size() == output_rank);
#endif
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (size_t i = 0; i < output_rank; ++i) {
    bool is_axis = false;
    for (int64_t axis : axes_) {
      if (i == axis) {
        is_axis = true;
        break;
      }
    }
    if (!is_axis) {
      exprs.push_back(builder.getAffineDimExpr(i));
    }
  }
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

std::unique_ptr<UnSqueezeKernelGenerator>
UnSqueezeKernelGenerator::Make(std::vector<int64_t> axes) {
  return std::make_unique<UnSqueezeKernelGeneratorImpl>(std::move(axes));
}

UnSqueezeKernelGeneratorImpl::UnSqueezeKernelGeneratorImpl(
    std::vector<int64_t> axes)
    : axes_(std::move(axes)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
UnSqueezeKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<UnSqueezeKernel>
UnSqueezeKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                    llvm::ArrayRef<size_t> output_layout) {
  std::vector<int64_t> axes = axes_;
  return std::make_shared<UnSqueezeKernel>(std::move(axes));
}

} // namespace kernel
} // namespace cpu_transformers
