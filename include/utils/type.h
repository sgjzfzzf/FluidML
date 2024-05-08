#ifndef CPU_TRANSFORMERS_UTILS_TYPE_H_
#define CPU_TRANSFORMERS_UTILS_TYPE_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include <cstddef>
#include <cstdint>

namespace cpu_transformers {
enum class Type { UNKNOWN, BOOL, INT64, FLOAT16, FLOAT32, FLOAT64 };

Type GetType(std::int32_t type);

Type GetType(mlir::Type type);

size_t GetSizeFromType(Type type);

const char *GetStringFromType(Type type);

mlir::Type GetMLIRType(Type type, mlir::OpBuilder &builder);

} // namespace cpu_transformers

#endif
