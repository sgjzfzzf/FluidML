#ifndef CPU_TRANSFORMERS_STRUCTURE_MEMORY_LINEAR_H_
#define CPU_TRANSFORMERS_STRUCTURE_MEMORY_LINEAR_H_

#include "structure/memory/plan.h"

namespace cpu_transformers {
namespace memory {

class LinearPlan : public Plan {
public:
  LinearPlan() = default;
  LinearPlan(const LinearPlan &) = delete;
  LinearPlan(LinearPlan &&) = default;
  ~LinearPlan() = default;
  Index Run(Infos &info) const override;
};

} // namespace memory
} // namespace cpu_transformers

#endif
