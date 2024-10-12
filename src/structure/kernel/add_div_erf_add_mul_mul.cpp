#include "structure/kernel/add_div_erf_add_mul_mul.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace kernel {

std::string AddDivErfAddMulMulKernel::GetKernelName() const {
  return kKernelName;
}

AddDivErfAddMulMulKernel::AddDivErfAddMulMulKernel(
    Tensor &&add0_weight, Type div_type, float64_t div_weight, Type add1_type,
    float64_t add1_weight, Type mul1_type, float64_t mul1_weight)
    : add0_weight_(std::move(add0_weight)), div_type_(div_type),
      div_weight_(div_weight), add1_type_(add1_type), add1_weight_(add1_weight),
      mul1_type_(mul1_type), mul1_weight_(mul1_weight) {}

void AddDivErfAddMulMulKernel::Run(mlir::OpBuilder &builder, mlir::Value &input,
                                   mlir::Value &output) const {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Type add0_weight_type;
  if (add0_weight_.GetType() == Type::kFloat32) {
    add0_weight_type = mlir::FloatType::getF32(context);
  } else {
#ifdef DEBUG
    assert(false && "unimplemented type");
#else
    __builtin_unreachable();
#endif
  }
  mlir::MemRefType input_type = mlir::cast<mlir::MemRefType>(input.getType()),
                   output_type = mlir::cast<mlir::MemRefType>(output.getType());
  const size_t rank = input_type.getRank();
#ifdef DEBUG
  assert(rank == output_type.getRank());
#endif
  mlir::RankedTensorType add0_weight_tensor_type =
      mlir::RankedTensorType::get(add0_weight_.GetShape(), add0_weight_type);
  const std::vector<float64_t> &add0_weight_data = add0_weight_.Get();
  mlir::DenseElementsAttr add0_weight_elements;
  if (add0_weight_.GetType() == Type::kFloat32) {
    llvm::SmallVector<float32_t> add0_weight(add0_weight_data.begin(),
                                             add0_weight_data.end());
    add0_weight_elements = mlir::DenseElementsAttr::get(
        add0_weight_tensor_type, llvm::ArrayRef(add0_weight));
  } else {
#ifdef DEBUG
    assert(false && "unimplemented type");
#else
    __builtin_unreachable();
#endif
  }
  mlir::Value add0_weight = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), add0_weight_elements);
  mlir::MemRefType add0_weight_memref_type =
      mlir::MemRefType::get(add0_weight_.GetShape(), add0_weight_type);
  mlir::Value add0_weight_ref = builder.create<mlir::bufferization::ToMemrefOp>(
      builder.getUnknownLoc(), add0_weight_memref_type, add0_weight);
  mlir::Type div_type;
  if (div_type_ == Type::kFloat32) {
    div_type = builder.getF32Type();
  } else {
#ifdef DEBUG
    assert(false && "unimplemented type");
#else
    __builtin_unreachable();
#endif
  }
  mlir::Value div_weight = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getFloatAttr(div_type, div_weight_));
  mlir::Type add1_type;
  if (add1_type_ == Type::kFloat32) {
    add1_type = builder.getF32Type();
  } else {
#ifdef DEBUG
    assert(false && "unimplemented type");
#else
    __builtin_unreachable();
#endif
  }
  mlir::Value add1_weight = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getFloatAttr(add1_type, add1_weight_));
  mlir::Type mul1_type;
  if (mul1_type_ == Type::kFloat32) {
    mul1_type = builder.getF32Type();
  } else {
#ifdef DEBUG
    assert(false && "unimplemented type");
#else
    __builtin_unreachable();
#endif
  }
  llvm::SmallVector<mlir::AffineMap> maps = getBroadcastAffineMaps(
      builder, llvm::ArrayRef{input_type, add0_weight_memref_type},
      output_type);
  mlir::Value mul1_weight = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getFloatAttr(mul1_type, mul1_weight_));
  builder.create<mlir::linalg::GenericOp>(
      builder.getUnknownLoc(), mlir::TypeRange{},
      mlir::ValueRange{input, add0_weight_ref}, mlir::ValueRange{output}, maps,
      llvm::SmallVector<mlir::utils::IteratorType>(
          rank, mlir::utils::IteratorType::parallel),
      [&](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
#ifdef DEBUG
        assert(args.size() == 3);
#endif
        mlir::Value input = args[0], add0_weight = args[1], output = args[2],
                    add0_op =
                        b.create<mlir::arith::AddFOp>(loc, input, add0_weight),
                    div_op =
                        b.create<mlir::arith::DivFOp>(loc, add0_op, div_weight),
                    erf_op = b.create<mlir::math::ErfOp>(loc, div_op),
                    add1_op =
                        b.create<mlir::arith::AddFOp>(loc, erf_op, add1_weight),
                    mul0_op =
                        b.create<mlir::arith::MulFOp>(loc, add0_op, add1_op),
                    mul1_op = b.create<mlir::arith::MulFOp>(loc, mul0_op,
                                                            mul1_weight);
        b.create<mlir::linalg::YieldOp>(loc, mul1_op);
      });
}

} // namespace kernel
} // namespace cpu_transformers
