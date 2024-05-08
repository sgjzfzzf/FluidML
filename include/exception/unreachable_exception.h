#ifndef CPU_TRANSFORMERS_EXCEPTION_UNREACHABLE_EXCEPTION_H_
#define CPU_TRANSFORMERS_EXCEPTION_UNREACHABLE_EXCEPTION_H_

#include <exception>

namespace cpu_transformers {
class UnreachableException : public std::exception {
public:
  UnreachableException() = default;
  const char *what() const noexcept override;
};
} // namespace cpu_transformers

#endif
