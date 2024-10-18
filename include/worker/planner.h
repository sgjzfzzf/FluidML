#ifndef CPU_TRANSFORMERS_WORKER_PLANNER_H_
#define CPU_TRANSFORMERS_WORKER_PLANNER_H_

#include "structure/context/context.h"
#include "structure/flow/flow.h"
#include "structure/flow/sequence.h"
#include "structure/memory/index.h"
#include "worker/fwd.h"
#include <memory>
#include <tuple>

namespace cpu_transformers {
namespace worker {

class Planner {
public:
  virtual ~Planner() = default;
  virtual std::tuple<flow::Sequence, memory::Index>
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
} // namespace cpu_transformers

#endif
