#ifndef CPU_TRANSFORMERS_STRUCTURE_MEMORY_PLAN_H_
#define CPU_TRANSFORMERS_STRUCTURE_MEMORY_PLAN_H_

#include "structure/memory/index.h"
#include "structure/memory/info.h"

namespace cpu_transformers {
namespace memory {

class Plan {
public:
  Plan() = default;
  Plan(const Plan &) = delete;
  Plan(Plan &&) = default;
  virtual ~Plan() = default;
  virtual Index Run(Infos &info) const = 0;
};

} // namespace memory
} // namespace cpu_transformers

#endif
