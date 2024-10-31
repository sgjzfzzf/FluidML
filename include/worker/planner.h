#ifndef FLUIDML_WORKER_PLANNER_H_
#define FLUIDML_WORKER_PLANNER_H_

#include "nlohmann/json.hpp"
#include "structure/context/context.h"
#include "structure/flow/flow.h"
#include "structure/flow/sequence.h"
#include "structure/memory/index.h"
#include "worker/fwd.h"
#include <memory>
#include <tuple>

namespace fluidml {
namespace worker {

class Planner {
public:
  virtual ~Planner() = default;
  virtual std::tuple<flow::Sequence, memory::Index, nlohmann::json>
  Run(const flow::Flow &flow) = 0;
  static std::unique_ptr<Planner>
  MakePlainLinearPlanner(context::Context &&context);
  static std::unique_ptr<Planner>
  MakePlainGreedyPlanner(context::Context &&context);
  static std::unique_ptr<Planner>
  MakeDPGreedyPlanner(context::Context &&context);

protected:
  Planner() = default;
  Planner(const Planner &) = delete;
  Planner(Planner &&) = default;
};

} // namespace worker
} // namespace fluidml

#endif
