#ifndef CPU_TRANSFORMERS_OPTIMIZATION_MANAGER_H_
#define CPU_TRANSFORMERS_OPTIMIZATION_MANAGER_H_

namespace cpu_transformers {
namespace optimization {

class PassesManager {
public:
  PassesManager() = default;
  PassesManager(const PassesManager &manager) = default;
  PassesManager(PassesManager &&manager) = default;
};

} // namespace optimization
} // namespace cpu_transformers

#endif
