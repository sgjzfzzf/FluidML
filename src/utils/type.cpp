#include "utils/type.h"
#include "mlir/IR/BuiltinTypes.h"
#include "utils/float.h"
#ifdef DEBUG
#include "exception/unreachable_exception.h"
#endif

// TODO: add more types
namespace cpu_transformers {
Type GetType(std::int32_t type) {
  switch (type) {
  case 1:
    return Type::kFloat32;
  case 7:
    return Type::kInt64;
  case 9:
    return Type::kBool;
  case 10:
    return Type::kFloat64;
  default:
    return Type::kUnknown;
  }
}

Type GetType(mlir::Type type) {
  if (mlir::isa<mlir::Float32Type>(type)) {
    return Type::kFloat32;
  } else if (mlir::isa<mlir::IntegerType>(type)) {
    return Type::kInt64;
  } else if (mlir::isa<mlir::Float64Type>(type)) {
    return Type::kFloat64;
  } else {
    return Type::kUnknown;
  }
}

size_t GetSizeFromType(Type type) {
  switch (type) {
  case Type::kBool:
    return sizeof(bool);
  case Type::kInt64:
    return sizeof(int64_t);
  case Type::kFloat32:
    return sizeof(float32_t);
  case Type::kFloat64:
    return sizeof(float64_t);
  default:
#ifdef DEBUG
    throw UnreachableException();
#else
    __builtin_unreachable();
#endif
  }
}

const char *GetStringFromType(Type type) {
  switch (type) {
  case Type::kBool:
    return "bool";
  case Type::kInt64:
    return "int64";
  case Type::kFloat32:
    return "float32";
  case Type::kFloat64:
    return "float64";
  default:
#ifdef DEBUG
    throw UnreachableException();
#else
    __builtin_unreachable();
#endif
  }
}

mlir::Type GetMLIRType(Type type, mlir::OpBuilder &builder) {
  switch (type) {
  case Type::kBool:
    return builder.getI1Type();
  case Type::kInt64:
    return builder.getI64Type();
  case Type::kFloat32:
    return builder.getF32Type();
  case Type::kFloat64:
    return builder.getF64Type();
  default:
#ifdef DEBUG
    throw UnreachableException();
#else
    __builtin_unreachable();
#endif
  }
}
} // namespace cpu_transformers