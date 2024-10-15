#include "structure/kernel/mul.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "structure/kernel/utils.h"
#include <memory>

namespace cpu_transformers {
namespace kernel {

class MulConstantKernelGeneratorImpl : public MulConstantKernelGenerator {
public:
  MulConstantKernelGeneratorImpl(Type type, float64_t constant);
  MulConstantKernelGeneratorImpl(
      const MulConstantKernelGeneratorImpl &generator) = delete;
  MulConstantKernelGeneratorImpl(MulConstantKernelGeneratorImpl &&generator) =
      default;
  virtual ~MulConstantKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<MulConstantKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const Type type_;
  const float64_t constant_;
};

class MulCommonKernelGeneratorImpl : public MulCommonKernelGenerator {
public:
  MulCommonKernelGeneratorImpl() = default;
  MulCommonKernelGeneratorImpl(const MulCommonKernelGeneratorImpl &generator) =
      delete;
  MulCommonKernelGeneratorImpl(MulCommonKernelGeneratorImpl &&generator) =
      default;
  virtual ~MulCommonKernelGeneratorImpl() = default;
  std::shared_ptr<DoubleInputsWithoutBufferKernel>
  YieldDoubleInputsWithoutBufferKernel(
      llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<MulCommonKernel>
  Yield(llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
        llvm::ArrayRef<size_t> output_layout) override;
};

MulConstantKernel::MulConstantKernel(Type type, float64_t constant)
    : type_(type), constant_(constant) {}

std::string MulConstantKernel::GetKernelName() const { return kKernelName; }

void MulConstantKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                            mlir::Value &output) const {
  mlir::Value constant;
  if (type_ == Type::kFloat32) {
    constant = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getF32FloatAttr(constant_));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  mlir::MemRefType lhs_type = mlir::cast<mlir::MemRefType>(lhs.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  size_t rank = lhs_type.getRank();
#ifdef DEBUG
  assert(rank == output_type.getRank());
#endif
  mlir::AffineMap lhs_map = builder.getMultiDimIdentityMap(rank),
                  output_map = builder.getMultiDimIdentityMap(rank);
  llvm::SmallVector<mlir::AffineMap> maps = {lhs_map, output_map};
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{lhs},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 2);
#endif
        mlir::Value lhs = inputs[0];
        mlir::Value mul_op = b.create<mlir::arith::MulFOp>(loc, lhs, constant);
        b.create<mlir::linalg::YieldOp>(loc, mul_op);
      });
}

void MulCommonKernel::Run(mlir::OpBuilder &builder, mlir::Value &lhs,
                          mlir::Value &rhs, mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType lhs_type = mlir::cast<mlir::MemRefType>(lhs.getType()),
                   rhs_type = mlir::cast<mlir::MemRefType>(rhs.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = output_type.getRank();
#ifdef DEBUG
  assert(rank <= lhs_type.getRank());
  assert(rank <= rhs_type.getRank());
#endif
  llvm::SmallVector<mlir::AffineMap> maps =
      GetBroadcastAffineMaps(builder, {lhs_type, rhs_type}, output_type);
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{output}, maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value lhs = inputs[0], rhs = inputs[1],
                    mul_op = b.create<mlir::arith::MulFOp>(loc, lhs, rhs);
        b.create<mlir::linalg::YieldOp>(loc, mul_op);
      });
}

std::string MulCommonKernel::GetKernelName() const { return kKernelName; }

std::unique_ptr<MulConstantKernelGenerator>
MulConstantKernelGenerator::Make(Type type, float64_t constant) {
  return std::make_unique<MulConstantKernelGeneratorImpl>(type, constant);
}

std::unique_ptr<MulCommonKernelGenerator> MulCommonKernelGenerator::Make() {
  return std::make_unique<MulCommonKernelGeneratorImpl>();
}

MulConstantKernelGeneratorImpl::MulConstantKernelGeneratorImpl(
    Type type, float64_t constant)
    : type_(type), constant_(constant) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
MulConstantKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<MulConstantKernel>
MulConstantKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                      llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<MulConstantKernel>(type_, constant_);
}

std::shared_ptr<DoubleInputsWithoutBufferKernel>
MulCommonKernelGeneratorImpl::YieldDoubleInputsWithoutBufferKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return std::make_unique<MulCommonKernel>();
}

std::shared_ptr<MulCommonKernel>
MulCommonKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> lhs_layout,
                                    llvm::ArrayRef<size_t> rhs_layout,
                                    llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<MulCommonKernel>();
}

} // namespace kernel
} // namespace cpu_transformers
