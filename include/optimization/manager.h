#ifndef FLUIDML_OPTIMIZATION_MANAGER_H_
#define FLUIDML_OPTIMIZATION_MANAGER_H_

namespace fluidml {
namespace optimization {

class PassesManager {
public:
  PassesManager() = default;
  PassesManager(const PassesManager &manager) = default;
  PassesManager(PassesManager &&manager) = default;
};

} // namespace optimization
} // namespace fluidml

#endif
