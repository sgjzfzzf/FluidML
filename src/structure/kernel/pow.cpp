#include "structure/kernel/pow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include <memory>

namespace cpu_transformers {
namespace kernel {

class PowKernelGeneratorImpl : public PowKernelGenerator {
public:
  PowKernelGeneratorImpl(Type type, float64_t exp);
  PowKernelGeneratorImpl(const PowKernelGeneratorImpl &generator) = delete;
  PowKernelGeneratorImpl(PowKernelGeneratorImpl &&generator) = default;
  virtual ~PowKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<PowKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const Type type_;
  const float64_t exp_;
};

PowKernel::PowKernel(Type type, float64_t exp) : type_(type), exp_(exp) {}

std::string PowKernel::GetKernelName() const { return kKernelName; }

void PowKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                    mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Value exp;
  if (type_ == Type::kFloat32) {
    exp = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getF32FloatAttr(exp_));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = input_type.getRank();
#ifdef DEBUG
  assert(rank == output_type.getRank());
  assert(input_type.getShape() == output_type.getShape());
#endif
  Type input_raw_type = GetType(input_type.getElementType()),
       output_raw_type = GetType(output_type.getElementType());
  mlir::AffineMap input_map = builder.getMultiDimIdentityMap(rank),
                  output_map = builder.getMultiDimIdentityMap(rank);
  llvm::SmallVector<mlir::AffineMap> maps = {input_map, output_map};
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types;
  for (size_t i = 0; i < rank; ++i) {
    iterator_types.push_back(mlir::utils::IteratorType::parallel);
  }
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{input},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value input = inputs[0], output = inputs[1], pow_op;
        if (input_raw_type == Type::kFloat32 && type_ == Type::kFloat32 &&
            output_raw_type == Type::kFloat32) {
          pow_op = b.create<mlir::math::PowFOp>(loc, input, exp);
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
        b.create<mlir::linalg::YieldOp>(loc, pow_op);
      });
}

std::unique_ptr<PowKernelGenerator> PowKernelGenerator::Make(Type type,
                                                             float64_t exp) {
  return std::make_unique<PowKernelGeneratorImpl>(type, exp);
}

PowKernelGeneratorImpl::PowKernelGeneratorImpl(Type type, float64_t exp)
    : type_(type), exp_(exp) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
PowKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<PowKernel>
PowKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                              llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<PowKernel>(type_, exp_);
}

} // namespace kernel
} // namespace cpu_transformers
