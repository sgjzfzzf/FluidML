#include "exception/unreachable_exception.h"

namespace cpu_transformers {
const char *UnreachableException::what() const noexcept {
  return "Reached an unreachable point in the code.";
}
} // namespace cpu_transformers
