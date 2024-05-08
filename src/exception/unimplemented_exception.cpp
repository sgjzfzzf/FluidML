#include "exception/unimplemented_exception.h"

namespace cpu_transformers {
const char *UnimplementedException::what() const noexcept {
  return "It's still unimplemented.";
}
} // namespace cpu_transformers
