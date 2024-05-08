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
    return Type::FLOAT32;
  case 7:
    return Type::INT64;
  case 9:
    return Type::BOOL;
  case 10:
    return Type::FLOAT64;
  default:
    return Type::UNKNOWN;
  }
}

Type GetType(mlir::Type type) {
  if (mlir::isa<mlir::Float32Type>(type)) {
    return Type::FLOAT32;
  } else if (mlir::isa<mlir::IntegerType>(type)) {
    return Type::INT64;
  } else if (mlir::isa<mlir::Float64Type>(type)) {
    return Type::FLOAT64;
  } else {
    return Type::UNKNOWN;
  }
}

size_t GetSizeFromType(Type type) {
  switch (type) {
  case Type::BOOL:
    return sizeof(bool);
  case Type::INT64:
    return sizeof(int64_t);
  case Type::FLOAT32:
    return sizeof(float32_t);
  case Type::FLOAT64:
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
  case Type::BOOL:
    return "bool";
  case Type::INT64:
    return "int64";
  case Type::FLOAT32:
    return "float32";
  case Type::FLOAT64:
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
  case Type::BOOL:
    return builder.getI1Type();
  case Type::INT64:
    return builder.getI64Type();
  case Type::FLOAT32:
    return builder.getF32Type();
  case Type::FLOAT64:
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