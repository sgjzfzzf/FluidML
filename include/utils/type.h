#ifndef FLUIDML_UTILS_TYPE_H_
#define FLUIDML_UTILS_TYPE_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include <cstddef>
#include <cstdint>

namespace fluidml {

enum class Type {
  kUnknown,
  kBool,
  kInt32,
  kInt64,
  kFloat16,
  kFloat32,
  kFloat64
};

Type GetType(std::int32_t type);

Type GetType(mlir::Type type);

size_t GetSize(Type type);

const char *GetStringFromType(Type type);

mlir::Type GetMLIRType(Type type, mlir::OpBuilder &builder);

} // namespace fluidml

#endif
