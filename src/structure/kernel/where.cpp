#include "structure/kernel/where.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "structure/kernel/utils.h"
#include "structure/tensor/tensor.h"
#include "utils/float.h"
#include "utils/type.h"
#include <cstdint>
#include <memory>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

class WhereConstantCondConstantScalarYKernelGeneratorImpl
    : public WhereConstantCondConstantScalarYKernelGenerator {
public:
  WhereConstantCondConstantScalarYKernelGeneratorImpl(Tensor &&cond, Type type,
                                                      float64_t y);
  WhereConstantCondConstantScalarYKernelGeneratorImpl(
      const WhereConstantCondConstantScalarYKernelGeneratorImpl &generator) =
      delete;
  WhereConstantCondConstantScalarYKernelGeneratorImpl(
      WhereConstantCondConstantScalarYKernelGeneratorImpl &&generator) =
      default;
  virtual ~WhereConstantCondConstantScalarYKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<WhereConstantCondConstantScalarYKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const Tensor cond_;
  const Type type_;
  const float64_t y_;
};

class WhereConstantCondConstantTensorYKernelGeneratorImpl
    : public WhereConstantCondConstantTensorYKernelGenerator {
public:
  WhereConstantCondConstantTensorYKernelGeneratorImpl(Tensor &&cond,
                                                      Tensor &&y);
  WhereConstantCondConstantTensorYKernelGeneratorImpl(
      const WhereConstantCondConstantTensorYKernelGeneratorImpl &generator) =
      delete;
  WhereConstantCondConstantTensorYKernelGeneratorImpl(
      WhereConstantCondConstantTensorYKernelGeneratorImpl &&generator) =
      default;
  virtual ~WhereConstantCondConstantTensorYKernelGeneratorImpl() = default;
  std::shared_ptr<SingleInputWithoutBufferKernel>
  YieldSingleInputWithoutBufferKernel(
      llvm::ArrayRef<size_t> input_layout,
      llvm::ArrayRef<size_t> output_layout) override;
  std::shared_ptr<WhereConstantCondConstantTensorYKernel>
  Yield(llvm::ArrayRef<size_t> input_layout,
        llvm::ArrayRef<size_t> output_layout) override;

private:
  const Tensor cond_;
  const Tensor y_;
};

WhereConstantCondConstantScalarYKernel::WhereConstantCondConstantScalarYKernel(
    Tensor &&cond, Type type, float64_t y)
    : cond_(std::move(cond)), type_(type), y_(y) {}

std::string WhereConstantCondConstantScalarYKernel::GetKernelName() const {
  return kKernelName;
}

void WhereConstantCondConstantScalarYKernel::Run(mlir::OpBuilder &builder,
                                                 mlir::Value &input,
                                                 mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  const std::vector<int64_t> &cond_shape = cond_.GetShape();
  const std::vector<float64_t> &cond_ref = cond_.Get();
  mlir::MemRefType x_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = output_type.getRank();
#ifdef DEBUG
  assert(cond_.GetType() == Type::kBool);
  // TODO: add support for other types in the future
  assert(type_ == Type::kFloat32);
#endif
  mlir::RankedTensorType cond_tensor_type =
      mlir::RankedTensorType::get(cond_shape, builder.getI1Type());
  mlir::MemRefType cond_memref_type =
      mlir::MemRefType::get(cond_shape, builder.getI1Type());
  llvm::SmallVector<mlir::APInt> cond_data;
  for (bool i : cond_ref) {
    cond_data.push_back(mlir::APInt(1, i, true));
  }
  mlir::DenseElementsAttr cond_elements =
      mlir::DenseElementsAttr::get(cond_tensor_type, cond_data);
  mlir::arith::ConstantOp cond_value = builder.create<mlir::arith::ConstantOp>(
                              builder.getUnknownLoc(), cond_elements),
                          y_value = builder.create<mlir::arith::ConstantOp>(
                              builder.getUnknownLoc(),
                              builder.getF32FloatAttr(y_));
  mlir::bufferization::ToMemrefOp cond_memref =
      builder.create<mlir::bufferization::ToMemrefOp>(
          builder.getUnknownLoc(), cond_memref_type, cond_value);
  llvm::SmallVector<mlir::AffineMap> maps = GetBroadcastAffineMaps(
      builder, llvm::ArrayRef<mlir::MemRefType>{cond_memref_type, x_type},
      output_type);
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{cond_memref, input}, mlir::ValueRange{output}, maps,
      iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 3);
#endif
        mlir::Value select = inputs[0], x = inputs[1],
                    select_op = b.create<mlir::arith::SelectOp>(loc, select, x,
                                                                y_value);
        b.create<mlir::linalg::YieldOp>(loc, select_op);
      });
}

WhereConstantCondConstantTensorYKernel::WhereConstantCondConstantTensorYKernel(
    Tensor &&cond, Tensor &&y)
    : cond_(std::move(cond)), y_(std::move(y)) {}

std::string WhereConstantCondConstantTensorYKernel::GetKernelName() const {
  return kKernelName;
}

void WhereConstantCondConstantTensorYKernel::Run(mlir::OpBuilder &builder,
                                                 mlir::Value &input,
                                                 mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  const std::vector<int64_t> &cond_shape = cond_.GetShape(),
                             &y_shape = y_.GetShape();
  const std::vector<float64_t> &cond_ref = cond_.Get(), &y_ref = y_.Get();
  mlir::MemRefType x_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = output_type.getRank();
#ifdef DEBUG
  assert(cond_.GetType() == Type::kBool);
  // TODO: add support for other types in the future
  assert(y_.GetType() == Type::kFloat32);
#endif
  llvm::SmallVector<mlir::APInt> cond_data;
  for (bool i : cond_ref) {
    cond_data.push_back(mlir::APInt(1, i, true));
  }
  mlir::RankedTensorType cond_tensor_type =
      mlir::RankedTensorType::get(cond_shape, builder.getI1Type());
  mlir::MemRefType cond_memref_type =
      mlir::MemRefType::get(cond_shape, builder.getI1Type());
  mlir::DenseElementsAttr cond_elements =
      mlir::DenseElementsAttr::get(cond_tensor_type, cond_data);
  mlir::arith::ConstantOp cond_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), cond_elements);
  mlir::bufferization::ToMemrefOp cond_memref =
      builder.create<mlir::bufferization::ToMemrefOp>(
          builder.getUnknownLoc(), cond_memref_type, cond_value);
  llvm::SmallVector<float32_t> y_data(y_ref.begin(), y_ref.end());
  mlir::RankedTensorType y_tensor_type =
      mlir::RankedTensorType::get(y_shape, builder.getF32Type());
  mlir::MemRefType y_memref_type =
      mlir::MemRefType::get(y_shape, builder.getF32Type());
  mlir::DenseElementsAttr y_elements =
      mlir::DenseElementsAttr::get(y_tensor_type, llvm::ArrayRef(y_data));
  mlir::arith::ConstantOp y_value = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), y_elements);
  mlir::bufferization::ToMemrefOp y_memref =
      builder.create<mlir::bufferization::ToMemrefOp>(builder.getUnknownLoc(),
                                                      y_memref_type, y_value);
  llvm::SmallVector<mlir::AffineMap> maps = GetBroadcastAffineMaps(
      builder, llvm::ArrayRef<mlir::MemRefType>{cond_memref_type, x_type},
      output_type);
  llvm::SmallVector<mlir::utils::IteratorType> iterator_types(
      rank, mlir::utils::IteratorType::parallel);
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{cond_memref, input, y_memref}, mlir::ValueRange{output},
      maps, iterator_types,
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange inputs) {
#ifdef DEBUG
        assert(inputs.size() == 4);
#endif
        mlir::Value select = inputs[0], x = inputs[1], y = inputs[2],
                    select_op =
                        b.create<mlir::arith::SelectOp>(loc, select, x, y);
        b.create<mlir::linalg::YieldOp>(loc, select_op);
      });
}

std::unique_ptr<WhereConstantCondConstantScalarYKernelGenerator>
WhereConstantCondConstantScalarYKernelGenerator::Make(Tensor &&cond, Type type,
                                                      float64_t y) {
  return std::make_unique<WhereConstantCondConstantScalarYKernelGeneratorImpl>(
      std::move(cond), type, y);
}

std::unique_ptr<WhereConstantCondConstantTensorYKernelGenerator>
WhereConstantCondConstantTensorYKernelGenerator::Make(Tensor &&cond,
                                                      Tensor &&y) {
  return std::make_unique<WhereConstantCondConstantTensorYKernelGeneratorImpl>(
      std::move(cond), std::move(y));
}

WhereConstantCondConstantScalarYKernelGeneratorImpl::
    WhereConstantCondConstantScalarYKernelGeneratorImpl(Tensor &&cond,
                                                        Type type, float64_t y)
    : cond_(std::move(cond)), type_(type), y_(y) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
WhereConstantCondConstantScalarYKernelGeneratorImpl::
    YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                        llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<WhereConstantCondConstantScalarYKernel>
WhereConstantCondConstantScalarYKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  Tensor cond = cond_;
  return std::make_shared<WhereConstantCondConstantScalarYKernel>(
      std::move(cond), type_, y_);
}

WhereConstantCondConstantTensorYKernelGeneratorImpl::
    WhereConstantCondConstantTensorYKernelGeneratorImpl(Tensor &&cond,
                                                        Tensor &&y)
    : cond_(std::move(cond)), y_(std::move(y)) {}

std::shared_ptr<SingleInputWithoutBufferKernel>
WhereConstantCondConstantTensorYKernelGeneratorImpl::
    YieldSingleInputWithoutBufferKernel(llvm::ArrayRef<size_t> input_layout,
                                        llvm::ArrayRef<size_t> output_layout) {
  return Yield(input_layout, output_layout);
}

std::shared_ptr<WhereConstantCondConstantTensorYKernel>
WhereConstantCondConstantTensorYKernelGeneratorImpl::Yield(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  Tensor cond = cond_, y = y_;
  return std::make_shared<WhereConstantCondConstantTensorYKernel>(
      std::move(cond), std::move(y));
}

} // namespace kernel
} // namespace cpu_transformers
