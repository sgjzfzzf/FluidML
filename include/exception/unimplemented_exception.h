#ifndef CPU_TRANSFORMERS_EXCEPTION_UNIMPLEMENTED_EXCEPTION_H_
#define CPU_TRANSFORMERS_EXCEPTION_UNIMPLEMENTED_EXCEPTION_H_

#include <exception>

namespace cpu_transformers {
class UnimplementedException : std::exception {
public:
  UnimplementedException() = default;
  const char *what() const noexcept override;
};
} // namespace cpu_transformers

#endif
