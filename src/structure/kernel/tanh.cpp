#include "structure/kernel/tanh.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"

namespace cpu_transformers {
namespace kernel {

class TanhKernelGeneratorImpl : public TanhKernelGenerator {
public:
  TanhKernelGeneratorImpl() = default;
  TanhKernelGeneratorImpl(const TanhKernelGeneratorImpl &generator) = delete;
  TanhKernelGeneratorImpl(TanhKernelGeneratorImpl &&generator) = default;
  virtual ~TanhKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<TanhKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
};

std::string TanhKernel::GetKernelName() const { return kKernelName; }

void TanhKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                     mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Value value;
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  size_t rank = input_type.getRank();
#ifdef DEBUG
  assert(rank == output_type.getRank());
  assert(input_type.getShape() == output_type.getShape());
#endif
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
        mlir::Value input = inputs[0], output = inputs[1],
                    tanh_op = b.create<mlir::math::TanhOp>(
                        builder.getUnknownLoc(), input);
        b.create<mlir::linalg::YieldOp>(loc, tanh_op);
      });
}

std::unique_ptr<TanhKernelGenerator> TanhKernelGenerator::Make() {
  return std::make_unique<TanhKernelGeneratorImpl>();
}

std::shared_ptr<SingleInputWithoutBufferKernel>
TanhKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<TanhKernel>
TanhKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                               llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<TanhKernel>();
}

} // namespace kernel
} // namespace cpu_transformers
