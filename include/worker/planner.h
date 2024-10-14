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
  virtual ~Planner() = default;

protected:
  Planner() = default;
  Planner(const Planner &) = delete;
  Planner(Planner &&) = default;
};

class ExecutionPlanner : virtual public Planner {
public:
  virtual ~ExecutionPlanner() = default;
  virtual flow::Sequence FlowToSequence(const flow::Flow &flow) const = 0;

protected:
  ExecutionPlanner() = default;
  ExecutionPlanner(const ExecutionPlanner &) = delete;
  ExecutionPlanner(ExecutionPlanner &&) = default;
  flow::Sequence topologicalSort(const flow::Flow &flow) const;
};

class PlainPlanner : public ExecutionPlanner {
public:
  virtual ~PlainPlanner() = default;
  flow::Sequence FlowToSequence(const flow::Flow &flow) const override;

protected:
  PlainPlanner() = default;
  PlainPlanner(const PlainPlanner &) = delete;
  PlainPlanner(PlainPlanner &&) = default;
};

class DynamicProgrammingPlanner : public ExecutionPlanner {
public:
  virtual ~DynamicProgrammingPlanner() = default;
  flow::Sequence FlowToSequence(const flow::Flow &flow) const override;

protected:
  DynamicProgrammingPlanner() = default;
  DynamicProgrammingPlanner(const DynamicProgrammingPlanner &) = delete;
  DynamicProgrammingPlanner(DynamicProgrammingPlanner &&) = default;
};

class MemoryPlanner : virtual public Planner {
public:
  virtual ~MemoryPlanner() = default;
  memory::Index Run(const flow::Sequence &sequence) const;

protected:
  MemoryPlanner(std::unique_ptr<memory::Plan> &&plan);
  MemoryPlanner(const MemoryPlanner &) = delete;
  MemoryPlanner(MemoryPlanner &&) = default;
  virtual std::unique_ptr<memory::Infos> createInfos() const = 0;
  std::unique_ptr<memory::Plan> plan_;
};

class LinearPlanner : public MemoryPlanner {
public:
  virtual ~LinearPlanner() = default;

protected:
  LinearPlanner();
  LinearPlanner(const LinearPlanner &) = delete;
  LinearPlanner(LinearPlanner &&) = default;
  std::unique_ptr<memory::Infos> createInfos() const override;
};

class GreedyPlanner : public MemoryPlanner {
public:
  virtual ~GreedyPlanner() = default;

protected:
  GreedyPlanner();
  GreedyPlanner(const GreedyPlanner &) = delete;
  GreedyPlanner(GreedyPlanner &&) = default;
  std::unique_ptr<memory::Infos> createInfos() const override;
};

class PlainLinearPlanner : public PlainPlanner, public LinearPlanner {
public:
  virtual ~PlainLinearPlanner() = default;
  static std::unique_ptr<PlainLinearPlanner> Make();

protected:
  PlainLinearPlanner() = default;
  PlainLinearPlanner(const PlainLinearPlanner &) = delete;
  PlainLinearPlanner(PlainLinearPlanner &&) = default;
};

class PlainGreedyPlanner : public PlainPlanner, public GreedyPlanner {
public:
  virtual ~PlainGreedyPlanner() = default;
  static std::unique_ptr<PlainGreedyPlanner> Make();

protected:
  PlainGreedyPlanner() = default;
  PlainGreedyPlanner(const PlainGreedyPlanner &) = delete;
  PlainGreedyPlanner(PlainGreedyPlanner &&) = default;
};

class DPGreedyPlanner : public DynamicProgrammingPlanner, public GreedyPlanner {
public:
  virtual ~DPGreedyPlanner() = default;
  static std::unique_ptr<DPGreedyPlanner> Make();

protected:
  DPGreedyPlanner() = default;
  DPGreedyPlanner(const DPGreedyPlanner &) = delete;
  DPGreedyPlanner(DPGreedyPlanner &&) = default;
};

} // namespace worker
} // namespace cpu_transformers

#endif
