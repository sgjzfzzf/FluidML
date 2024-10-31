#include "utils/type.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "utils/float.h"

// TODO: add more types
namespace fluidml {
Type GetType(std::int32_t type) {
  switch (type) {
  case 1:
    return Type::kFloat32;
  case 6:
    return Type::kInt32;
  case 7:
    return Type::kInt64;
  case 9:
    return Type::kBool;
  case 10:
    return Type::kFloat16;
  default:
    return Type::kUnknown;
  }
}

Type GetType(mlir::Type type) {
  if (type.isSignedInteger(1)) {
    return Type::kBool;
  } else if (type.isSignedInteger(32)) {
    return Type::kInt32;
  } else if (type.isSignedInteger(64)) {
    return Type::kInt64;
  } else if (type.isF16()) {
    return Type::kFloat16;
  } else if (type.isF32()) {
    return Type::kFloat32;
  } else if (type.isF64()) {
    return Type::kFloat64;
  } else {
    return Type::kUnknown;
  }
}

size_t GetSize(Type type) {
  switch (type) {
  case Type::kBool:
    return sizeof(bool);
  case Type::kInt32:
    return sizeof(int32_t);
  case Type::kInt64:
    return sizeof(int64_t);
  case Type::kFloat32:
    return sizeof(float32_t);
  case Type::kFloat64:
    return sizeof(float64_t);
  default:
#ifdef DEBUG
    assert(false && "unimplemented");
#else
    __builtin_unreachable();
#endif
  }
}

const char *GetStringFromType(Type type) {
  switch (type) {
  case Type::kBool:
    return "bool";
  case Type::kInt32:
    return "int32";
  case Type::kInt64:
    return "int64";
  case Type::kFloat32:
    return "float32";
  case Type::kFloat64:
    return "float64";
  default:
#ifdef DEBUG
    assert(false && "unimplemented");
#else
    __builtin_unreachable();
#endif
  }
}

mlir::Type GetMLIRType(Type type, mlir::OpBuilder &builder) {
  switch (type) {
  case Type::kBool:
    return builder.getI1Type();
  case Type::kInt32:
    return builder.getI32Type();
  case Type::kInt64:
    return builder.getI64Type();
  case Type::kFloat16:
    return builder.getF16Type();
  case Type::kFloat32:
    return builder.getF32Type();
  case Type::kFloat64:
    return builder.getF64Type();
  default:
#ifdef DEBUG
    assert(false && "unimplemented");
#else
    __builtin_unreachable();
#endif
  }
}
} // namespace fluidml