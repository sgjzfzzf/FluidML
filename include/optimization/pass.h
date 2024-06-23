#ifndef CPU_TRANSFORMERS_OPTIMIZATION_PASS_H_
#define CPU_TRANSFORMERS_OPTIMIZATION_PASS_H_

namespace cpu_transformers {
namespace optimization {

class Pass {
public:
  Pass() = default;
  Pass(const Pass &pass) = default;
  Pass(Pass &&pass) = default;
};

} // namespace optimization
} // namespace cpu_transformers

#endif
