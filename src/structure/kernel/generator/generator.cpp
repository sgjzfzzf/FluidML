#include "structure/kernel/generator/generator.h"

namespace cpu_transformers {
namespace kernel {

static constexpr size_t kHashSeed = 0x9e3779b9;

size_t SingleInputKernelGenerator::GetShapeHashCode() const {
  const Meta &input_meta = GetInputMeta(), output_meta = GetOutputMeta();
  const Type input_type = input_meta.GetType(),
             output_type = output_meta.GetType();
  const std::vector<int64_t> &input_shape = input_meta.GetShape(),
                             &output_shape = output_meta.GetShape();
  size_t hash = typeid(*this).hash_code();
  std::hash<Type> type_hasher;
  std::hash<int64_t> dim_hasher;
  hash ^= type_hasher(input_type) + kHashSeed + (hash << 6) + (hash >> 2);
  for (int64_t dim : input_shape) {
    hash ^= dim_hasher(dim) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  hash ^= type_hasher(output_type) + kHashSeed + (hash << 6) + (hash >> 2);
  for (int64_t dim : output_shape) {
    hash ^= dim_hasher(dim) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  return hash;
}

bool SingleInputKernelGenerator::ShapeEquals(
    const KernelGenerator &other) const {
  if (const SingleInputKernelGenerator *other_generator =
          dynamic_cast<const SingleInputKernelGenerator *>(&other)) {
    return ShapeEquals(*other_generator);
  } else {
    return false;
  }
}

bool SingleInputKernelGenerator::ShapeEquals(
    const SingleInputKernelGenerator &other) const {
  return GetInputMeta() == other.GetInputMeta() &&
         GetOutputMeta() == other.GetOutputMeta();
}

size_t DoubleInputsKernelGenerator::GetShapeHashCode() const {
  const Meta &lhs_meta = GetLhsMeta(), rhs_meta = GetRhsMeta(),
             output_meta = GetOutputMeta();
  const Type lhs_type = lhs_meta.GetType(), rhs_type = rhs_meta.GetType(),
             output_type = output_meta.GetType();
  const std::vector<int64_t> &lhs_shape = lhs_meta.GetShape(),
                             &rhs_shape = rhs_meta.GetShape(),
                             &output_shape = output_meta.GetShape();
  size_t hash = typeid(*this).hash_code();
  std::hash<Type> type_hasher;
  std::hash<int64_t> dim_hasher;
  hash ^= type_hasher(lhs_type) + kHashSeed + (hash << 6) + (hash >> 2);
  for (int64_t dim : lhs_shape) {
    hash ^= dim_hasher(dim) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  hash ^= type_hasher(rhs_type) + kHashSeed + (hash << 6) + (hash >> 2);
  for (int64_t dim : rhs_shape) {
    hash ^= dim_hasher(dim) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  hash ^= type_hasher(output_type) + kHashSeed + (hash << 6) + (hash >> 2);
  for (int64_t dim : output_shape) {
    hash ^= dim_hasher(dim) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  return hash;
}

bool DoubleInputsKernelGenerator::ShapeEquals(
    const KernelGenerator &other) const {
  if (const DoubleInputsKernelGenerator *other_generator =
          dynamic_cast<const DoubleInputsKernelGenerator *>(&other)) {
    return ShapeEquals(*other_generator);
  } else {
    return false;
  }
}

bool DoubleInputsKernelGenerator::ShapeEquals(
    const DoubleInputsKernelGenerator &other) const {
  return GetLhsMeta() == other.GetLhsMeta() &&
         GetRhsMeta() == other.GetRhsMeta() &&
         GetOutputMeta() == other.GetOutputMeta();
}

std::shared_ptr<SingleInputKernel>
SingleInputWithoutBufferKernelGenerator::YieldSingleInputKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return YieldSingleInputWithoutBufferKernel(input_layout, output_layout);
}

std::shared_ptr<SingleInputKernel>
SingleInputWithBufferKernelGenerator::YieldSingleInputKernel(
    llvm::ArrayRef<size_t> input_layout, llvm::ArrayRef<size_t> output_layout) {
  return YieldSingleInputWithBufferKernel(input_layout, output_layout);
}

std::shared_ptr<DoubleInputsKernel>
DoubleInputsWithoutBufferKernelGenerator::YieldDoubleInputsKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return YieldDoubleInputsWithoutBufferKernel(lhs_layout, rhs_layout,
                                              output_layout);
}

std::shared_ptr<DoubleInputsKernel>
DoubleInputsWithBufferKernelGenerator::YieldDoubleInputsKernel(
    llvm::ArrayRef<size_t> lhs_layout, llvm::ArrayRef<size_t> rhs_layout,
    llvm::ArrayRef<size_t> output_layout) {
  return YieldDoubleInputsWithBufferKernel(lhs_layout, rhs_layout,
                                           output_layout);
}

} // namespace kernel
} // namespace cpu_transformers
