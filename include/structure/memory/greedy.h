#ifndef CPU_TRANSFORMERS_STRUCTURE_MEMORY_GREEDY_H_
#define CPU_TRANSFORMERS_STRUCTURE_MEMORY_GREEDY_H_

#include "structure/memory/info.h"
#include "structure/memory/plan.h"

namespace cpu_transformers {
namespace memory {

class GreedyPlan : public Plan {
public:
  GreedyPlan() = default;
  GreedyPlan(const GreedyPlan &) = delete;
  GreedyPlan(GreedyPlan &&) = default;
  ~GreedyPlan() = default;
  Index Run(Infos &info) const override;
};

} // namespace memory
} // namespace cpu_transformers

#endif
