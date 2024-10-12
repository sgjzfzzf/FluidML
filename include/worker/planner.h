#ifndef CPU_TRANSFORMERS_WORKER_PLANNER_H_
#define CPU_TRANSFORMERS_WORKER_PLANNER_H_

#include "structure/flow/flow.h"
#include "structure/flow/sequence.h"
#include "structure/memory/index.h"
#include "structure/memory/plan.h"
#include "worker/fwd.h"
#include <memory>

namespace cpu_transformers {
namespace worker {

class Planner {
public:
  Planner() = default;
  Planner(const Planner &) = delete;
  Planner(Planner &&) = default;
  virtual ~Planner() = default;
};

class ExecutionPlanner : virtual public Planner {
public:
  ExecutionPlanner() = default;
  ExecutionPlanner(const ExecutionPlanner &) = delete;
  ExecutionPlanner(ExecutionPlanner &&) = default;
  virtual ~ExecutionPlanner() = default;
  virtual flow::Sequence FlowToSequence(const flow::Flow &flow) const = 0;

protected:
  flow::Sequence topologicalSort(const flow::Flow &flow) const;
};

class PlainPlanner : public ExecutionPlanner {
public:
  PlainPlanner() = default;
  PlainPlanner(const PlainPlanner &) = delete;
  PlainPlanner(PlainPlanner &&) = default;
  virtual ~PlainPlanner() = default;
  flow::Sequence FlowToSequence(const flow::Flow &flow) const override;
};

class DynamicProgrammingPlanner : public ExecutionPlanner {
public:
  DynamicProgrammingPlanner() = default;
  DynamicProgrammingPlanner(const DynamicProgrammingPlanner &) = delete;
  DynamicProgrammingPlanner(DynamicProgrammingPlanner &&) = default;
  virtual ~DynamicProgrammingPlanner() = default;
  flow::Sequence FlowToSequence(const flow::Flow &flow) const override;
};

class MemoryPlanner : virtual public Planner {
public:
  MemoryPlanner(std::unique_ptr<memory::Plan> &&plan);
  MemoryPlanner(const MemoryPlanner &) = delete;
  MemoryPlanner(MemoryPlanner &&) = default;
  virtual ~MemoryPlanner() = default;
  memory::Index Run(const flow::Sequence &sequence) const;

protected:
  virtual std::unique_ptr<memory::Infos> createInfos() const = 0;
  std::unique_ptr<memory::Plan> plan_;
};

class LinearPlanner : public MemoryPlanner {
public:
  LinearPlanner();
  LinearPlanner(const LinearPlanner &) = delete;
  LinearPlanner(LinearPlanner &&) = default;
  virtual ~LinearPlanner() = default;

private:
  std::unique_ptr<memory::Infos> createInfos() const override;
};

class GreedyPlanner : public MemoryPlanner {
public:
  GreedyPlanner();
  GreedyPlanner(const GreedyPlanner &) = delete;
  GreedyPlanner(GreedyPlanner &&) = default;
  virtual ~GreedyPlanner() = default;

private:
  std::unique_ptr<memory::Infos> createInfos() const override;
};

class PlainLinearPlanner : public PlainPlanner, public LinearPlanner {
public:
  PlainLinearPlanner() = default;
  PlainLinearPlanner(const PlainLinearPlanner &) = delete;
  PlainLinearPlanner(PlainLinearPlanner &&) = default;
  virtual ~PlainLinearPlanner() = default;
};

class PlainGreedyPlanner : public PlainPlanner, public GreedyPlanner {
public:
  PlainGreedyPlanner() = default;
  PlainGreedyPlanner(const PlainGreedyPlanner &) = delete;
  PlainGreedyPlanner(PlainGreedyPlanner &&) = default;
  virtual ~PlainGreedyPlanner() = default;
};

class DPGreedyPlanner : public DynamicProgrammingPlanner, public GreedyPlanner {
public:
  DPGreedyPlanner() = default;
  DPGreedyPlanner(const DPGreedyPlanner &) = delete;
  DPGreedyPlanner(DPGreedyPlanner &&) = default;
  virtual ~DPGreedyPlanner() = default;
};

} // namespace worker
} // namespace cpu_transformers

#endif
