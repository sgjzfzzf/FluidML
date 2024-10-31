#ifndef FLUIDML_STRUCTURE_MEMORY_PLAN_H_
#define FLUIDML_STRUCTURE_MEMORY_PLAN_H_

#include "structure/memory/index.h"
#include "structure/memory/info.h"

namespace fluidml {
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
} // namespace fluidml

#endif
