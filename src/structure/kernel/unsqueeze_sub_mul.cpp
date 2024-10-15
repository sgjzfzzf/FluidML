#include "structure/kernel/unsqueeze_sub_mul.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace cpu_transformers {
namespace kernel {

class UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl
    : public UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator {
public:
  UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl(
      std::vector<int64_t> &&unsqueeze_axes, const Type &sub_type,
      float64_t sub_val, const Type &mul_type, float64_t mul_val);
  UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl(
      const UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl &generator) =
      delete;
  UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl(
      UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl &&generator) =
      default;
  virtual ~UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<UnsqueezeSubLhsScalarMulRhsScalarKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const std::vector<int64_t> unsqueeze_axes_;
  const Type sub_type_;
  const float64_t sub_val_;
  const Type mul_type_;
  const float64_t mul_val_;
};

UnsqueezeSubLhsScalarMulRhsScalarKernel::
    UnsqueezeSubLhsScalarMulRhsScalarKernel(
        std::vector<int64_t> &&unsqueeze_axes, const Type &sub_type,
        float64_t sub_val, const Type &mul_type, float64_t mul_val)
    : unsqueeze_axes_(std::move(unsqueeze_axes)), sub_type_(sub_type),
      sub_val_(sub_val), mul_type_(mul_type), mul_val_(mul_val) {}

std::string UnsqueezeSubLhsScalarMulRhsScalarKernel::GetKernelName() const {
  return kKernelName;
}

void UnsqueezeSubLhsScalarMulRhsScalarKernel::Run(mlir::OpBuilder &builder,
                                                  mlir::Value &input,
                                                  mlir::Value &output) const {
// TODO: Only support for fp32 currently
#ifdef DEBUG
  assert(sub_type_ == Type::kFloat32);
  assert(mul_type_ == Type::kFloat32);
#endif
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t output_rank = output_type.getRank();
#ifdef DEBUG
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType());
  const size_t input_rank = input_type.getRank();
  assert(input_rank + unsqueeze_axes_.size() == output_rank);
#endif
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (size_t i = 0; i < output_rank; ++i) {
    bool is_axis = false;
    for (int64_t axis : unsqueeze_axes_) {
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
        mlir::Value input = inputs[0],
                    sub = b.create<mlir::arith::ConstantOp>(
                        loc, builder.getF32FloatAttr(sub_val_)),
                    mul = b.create<mlir::arith::ConstantOp>(
                        loc, builder.getF32FloatAttr(mul_val_)),
                    sub_op = b.create<mlir::arith::SubFOp>(loc, sub, input),
                    mul_op = b.create<mlir::arith::MulFOp>(loc, sub_op, mul);
        b.create<mlir::linalg::YieldOp>(loc, mul_op);
      });
}

std::unique_ptr<UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator>
UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator::Make(
    std::vector<int64_t> &&unsqueeze_axes, const Type &sub_type,
    float64_t sub_val, const Type &mul_type, float64_t mul_val) {
  return std::make_unique<UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl>(
      std::move(unsqueeze_axes), sub_type, sub_val, mul_type, mul_val);
}

UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl::
    UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl(
        std::vector<int64_t> &&unsqueeze_axes, const Type &sub_type,
        float64_t sub_val, const Type &mul_type, float64_t mul_val)
    : unsqueeze_axes_(std::move(unsqueeze_axes)), sub_type_(sub_type),
      sub_val_(sub_val), mul_type_(mul_type), mul_val_(mul_val) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl::
    YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                        llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<UnsqueezeSubLhsScalarMulRhsScalarKernel>
UnsqueezeSubLhsScalarMulRhsScalarKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  std::vector<int64_t> unsqueeze_axes = unsqueeze_axes_;
  return std::make_shared<UnsqueezeSubLhsScalarMulRhsScalarKernel>(
      std::move(unsqueeze_axes), sub_type_, sub_val_, mul_type_, mul_val_);
}

} // namespace kernel
} // namespace cpu_transformers
