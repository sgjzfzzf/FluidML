#ifndef CPU_TRANSFORMERS_WORKER_PLANNER_H_
#define CPU_TRANSFORMERS_WORKER_PLANNER_H_

#include "structure/flow/flow.h"
#include "structure/flow/sequence.h"
#include "structure/memory/index.h"
#include "structure/memory/plan.h"
#include <memory>

namespace cpu_transformers {
namespace worker {
class Planner {
public:
  Planner(std::unique_ptr<memory::Plan> &&plan);
  Planner(const Planner &) = delete;
  Planner(Planner &&) = default;
  virtual ~Planner() = default;
  flow::Sequence FlowToSequence(const flow::Flow &flow) const;
  memory::Index Run(const flow::Sequence &sequence) const;

private:
  virtual std::unique_ptr<memory::Infos> createInfos() const = 0;
  std::unique_ptr<memory::Plan> plan_;
};

class LinearPlanner : public Planner {
public:
  LinearPlanner();
  LinearPlanner(const LinearPlanner &) = delete;
  LinearPlanner(LinearPlanner &&) = default;
  ~LinearPlanner() = default;

private:
  std::unique_ptr<memory::Infos> createInfos() const override;
};

class GreedyPlanner : public Planner {
public:
  GreedyPlanner();
  GreedyPlanner(const GreedyPlanner &) = delete;
  GreedyPlanner(GreedyPlanner &&) = default;
  ~GreedyPlanner() = default;

private:
  std::unique_ptr<memory::Infos> createInfos() const override;
};

} // namespace worker
} // namespace cpu_transformers

#endif
