#include "structure/kernel/div.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "utils/type.h"
#include <memory>

namespace cpu_transformers {
namespace kernel {

class DivConstantRhsKernelGeneratorImpl : public DivConstantRhsKernelGenerator {
public:
  DivConstantRhsKernelGeneratorImpl(Type type, float64_t constant);
  DivConstantRhsKernelGeneratorImpl(
      const DivConstantRhsKernelGeneratorImpl &generator) = delete;
  DivConstantRhsKernelGeneratorImpl(
      DivConstantRhsKernelGeneratorImpl &&generator) = default;
  virtual ~DivConstantRhsKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<DivConstantRhsKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const Type type_;
  const float64_t constant_;
};

DivConstantRhsKernel::DivConstantRhsKernel(Type type, float64_t constant)
    : type_(type), constant_(constant) {}

std::string DivConstantRhsKernel::GetKernelName() const { return kKernelName; }

void DivConstantRhsKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                               mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
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
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  size_t rank = input_type.getRank();
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
        mlir::Value input = inputs[0], output = inputs[1], div_op;
        if (input_raw_type == Type::kFloat32 && type_ == Type::kFloat32 &&
            output_raw_type == Type::kFloat32) {
          div_op = b.create<mlir::arith::DivFOp>(loc, input, constant);
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
        b.create<mlir::linalg::YieldOp>(loc, div_op);
      });
}

std::unique_ptr<SingleInputWithoutBufferKernelGenerator>
DivConstantRhsKernelGenerator::Make(Type type, float64_t constant) {
  return std::make_unique<DivConstantRhsKernelGeneratorImpl>(type, constant);
}

DivConstantRhsKernelGeneratorImpl::DivConstantRhsKernelGeneratorImpl(
    Type type, float64_t constant)
    : type_(type), constant_(constant) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
DivConstantRhsKernelGeneratorImpl::YieldSingleInputWithoutBufferKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<DivConstantRhsKernel>
DivConstantRhsKernelGeneratorImpl::Yield(llvm::ArrayRef<size_t> input_layout,
                                         llvm::ArrayRef<size_t> output_layout) {
  return std::make_shared<DivConstantRhsKernel>(type_, constant_);
}

} // namespace kernel
} // namespace cpu_transformers
