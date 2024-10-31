#ifndef FLUIDML_OPTIMIZATION_PASS_H_
#define FLUIDML_OPTIMIZATION_PASS_H_

namespace fluidml {
namespace optimization {

class Pass {
public:
  Pass() = default;
  Pass(const Pass &pass) = default;
  Pass(Pass &&pass) = default;
};

} // namespace optimization
} // namespace fluidml

#endif
