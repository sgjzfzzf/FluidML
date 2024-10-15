#include "structure/kernel/erf.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

class ErfKernelGeneratorImpl : public ErfKernelGenerator {
public:
  ErfKernelGeneratorImpl() = default;
  ErfKernelGeneratorImpl(const ErfKernelGeneratorImpl &generator) = delete;
  ErfKernelGeneratorImpl(ErfKernelGeneratorImpl &&generator) = default;
  virtual ~ErfKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<ErfKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;
};

std::string ErfKernel::GetKernelName() const { return kKernelName; }

void ErfKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                    mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType());
  const size_t rank = input_type.getRank();
#ifdef DEBUG
  mlir::MemRefType output_type = mlir::cast<mlir::MemRefType>(output.getType());
  assert(rank == output_type.getRank());
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
        mlir::Value input = inputs[0],
                    erf_op = b.create<mlir::math::ErfOp>(loc, input);
        b.create<mlir::linalg::YieldOp>(loc, erf_op);
      });
}

std::unique_ptr<ErfKernelGenerator> ErfKernelGenerator::Make() {
  return std::make_unique<ErfKernelGeneratorImpl>();
}

std::shared_ptr<SingleInputWithoutBufferKernel>
ErfKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<ErfKernel>
ErfKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                              llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<ErfKernel>();
}

} // namespace kernel
} // namespace cpu_transformers
