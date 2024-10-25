#include "worker/planner.h"
#include "evaluation/dp.h"
#include "structure/context/context.h"
#include "structure/flow/edge.h"
#include "structure/flow/node.h"
#include "structure/flow/region.h"
#include "structure/flow/sequence.h"
#include "structure/memory/greedy.h"
#include "structure/memory/index.h"
#include "structure/memory/info.h"
#include "structure/memory/linear.h"
#include "utils/isa.hpp"
#include <memory>
#include <unordered_map>
#ifdef DEBUG
#include <cassert>
#endif

namespace {

using namespace cpu_transformers;
flow::Sequence TopologicalSort(
    const flow::Flow
        &flow) { // Run the topological sort algorithm to get the sequence.
  flow::Sequence sequence;
  std::unordered_map<std::string, std::shared_ptr<flow::Node>> unvisited_nodes;
  std::list<std::shared_ptr<flow::Node>> waiting_nodes;
  std::vector<std::shared_ptr<flow::Node>> nodes = flow.GetNodes();
  std::vector<std::shared_ptr<flow::Edge>> edges = flow.GetEdges();
  for (std::shared_ptr<flow::Node> &node : nodes) {
    std::string name = node->GetName();
    if (std::shared_ptr<flow::SingleInputNode> single_input_node =
            std::dynamic_pointer_cast<flow::SingleInputNode>(node)) {
      std::shared_ptr<flow::OwnToEdge> input_edge =
          flow.GetInputEdge(*single_input_node);
      if (isa<flow::InputEdge>(input_edge) ||
          isa<flow::ConstantEdge>(input_edge)) {
        waiting_nodes.push_back(std::move(node));
      } else {
        unvisited_nodes.insert({std::move(name), std::move(node)});
      }
    } else if (std::shared_ptr<flow::DoubleInputsNode> double_inputs_node =
                   std::dynamic_pointer_cast<flow::DoubleInputsNode>(node)) {
      std::shared_ptr<flow::OwnToEdge> lhs_edge =
                                           flow.GetLhsEdge(*double_inputs_node),
                                       rhs_edge =
                                           flow.GetRhsEdge(*double_inputs_node);
      if ((isa<flow::InputEdge>(lhs_edge) ||
           isa<flow::ConstantEdge>(lhs_edge)) &&
          (isa<flow::InputEdge>(rhs_edge) ||
           isa<flow::ConstantEdge>(rhs_edge))) {
        waiting_nodes.push_back(std::move(node));
      } else {
        unvisited_nodes.insert({std::move(name), std::move(node)});
      }
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
  }
  std::unordered_multimap<std::string, std::string> prev_nodes;
  for (std::shared_ptr<flow::Edge> edge : edges) {
    if (std::shared_ptr<flow::MemoryEdge> memory_edge =
            std::dynamic_pointer_cast<flow::MemoryEdge>(edge)) {
      std::shared_ptr<flow::Node> from = memory_edge->GetFrom(),
                                  to = memory_edge->GetTo();
      const std::string &from_name = from->GetName(), &to_name = to->GetName();
      prev_nodes.insert({std::move(to_name), std::move(from_name)});
    }
  }
  while (!waiting_nodes.empty()) {
    std::shared_ptr<flow::Node> node = std::move(waiting_nodes.back());
    const std::string &name = node->GetName();
    waiting_nodes.pop_back();
    sequence.PutNode(std::move(node));
    for (auto prev_it = prev_nodes.begin(); prev_it != prev_nodes.end();) {
      if (prev_it->second == name) {
        std::string prev_name = prev_it->first;
        prev_it = prev_nodes.erase(prev_it);
        if (prev_nodes.find(prev_name) == prev_nodes.end()) {
          auto it = unvisited_nodes.find(prev_name);
#ifdef DEBUG
          assert(it != unvisited_nodes.end());
#endif
          waiting_nodes.push_back(std::move(it->second));
          unvisited_nodes.erase(it);
        }
      } else {
        ++prev_it;
      }
    }
  }
#ifdef DEBUG
  assert(waiting_nodes.empty());
  assert(unvisited_nodes.empty());
#endif
  for (std::shared_ptr<flow::Edge> edge : edges) {
    sequence.PutEdge(std::move(edge));
  }
  const std::vector<std::shared_ptr<flow::Region>> &regions = flow.GetRegions();
  for (std::shared_ptr<flow::Region> region : regions) {
    sequence.PutRegion(std::move(region));
  }
  return sequence;
}

} // namespace

namespace cpu_transformers {
namespace worker {

class PlannerImpl : public Planner {
public:
  virtual ~PlannerImpl() = default;
  std::tuple<flow::Sequence, memory::Index, nlohmann::json>
  Run(const flow::Flow &flow) override;

protected:
  PlannerImpl(context::Context &&context);
  PlannerImpl(const PlannerImpl &) = delete;
  PlannerImpl(PlannerImpl &&) = default;
  virtual std::tuple<flow::Sequence, nlohmann::json>
  transform(const flow::Flow &flow) = 0;
  virtual std::unique_ptr<memory::Infos>
  makeInfos(const flow::Sequence &sequence) = 0;
  virtual std::unique_ptr<memory::Plan> makePlan() = 0;
  memory::Index run(const flow::Sequence &sequence);
  context::Context context_;
};

class PlainPlannerImpl : virtual public PlannerImpl {
public:
  virtual ~PlainPlannerImpl() = default;

protected:
  using PlannerImpl::PlannerImpl;
  PlainPlannerImpl(const PlainPlannerImpl &) = delete;
  PlainPlannerImpl(PlainPlannerImpl &&) = default;
  std::tuple<flow::Sequence, nlohmann::json>
  transform(const flow::Flow &flow) override;
};

class DynamicProgrammingPlannerImpl : virtual public PlannerImpl {
public:
  virtual ~DynamicProgrammingPlannerImpl() = default;

protected:
  using PlannerImpl::PlannerImpl;
  DynamicProgrammingPlannerImpl(const DynamicProgrammingPlannerImpl &) = delete;
  DynamicProgrammingPlannerImpl(DynamicProgrammingPlannerImpl &&) = default;
  std::tuple<flow::Sequence, nlohmann::json>
  transform(const flow::Flow &flow) override;
};

class LinearPlannerImpl : virtual public PlannerImpl {
public:
  virtual ~LinearPlannerImpl() = default;

protected:
  using PlannerImpl::PlannerImpl;
  LinearPlannerImpl(const LinearPlannerImpl &) = delete;
  LinearPlannerImpl(LinearPlannerImpl &&) = default;
  std::unique_ptr<memory::Infos>
  makeInfos(const flow::Sequence &sequence) override;
  std::unique_ptr<memory::Plan> makePlan() override;
};

class GreedyPlannerImpl : virtual public PlannerImpl {
public:
  virtual ~GreedyPlannerImpl() = default;

protected:
  using PlannerImpl::PlannerImpl;
  GreedyPlannerImpl(const GreedyPlannerImpl &) = delete;
  GreedyPlannerImpl(GreedyPlannerImpl &&) = default;
  std::unique_ptr<memory::Infos>
  makeInfos(const flow::Sequence &sequence) override;
  std::unique_ptr<memory::Plan> makePlan() override;
};

class PlainLinearPlannerImpl : public PlainPlannerImpl,
                               public LinearPlannerImpl {
public:
  PlainLinearPlannerImpl(context::Context &&context);
  PlainLinearPlannerImpl(const PlainLinearPlannerImpl &) = delete;
  PlainLinearPlannerImpl(PlainLinearPlannerImpl &&) = default;
  virtual ~PlainLinearPlannerImpl() = default;
};

class PlainGreedyPlannerImpl : public PlainPlannerImpl,
                               public GreedyPlannerImpl {
public:
  PlainGreedyPlannerImpl(context::Context &&context);
  PlainGreedyPlannerImpl(const PlainGreedyPlannerImpl &) = delete;
  PlainGreedyPlannerImpl(PlainGreedyPlannerImpl &&) = default;
  virtual ~PlainGreedyPlannerImpl() = default;
};

class DPGreedyPlannerImpl : public DynamicProgrammingPlannerImpl,
                            public GreedyPlannerImpl {
public:
  DPGreedyPlannerImpl(context::Context &&context);
  DPGreedyPlannerImpl(const DPGreedyPlannerImpl &) = delete;
  DPGreedyPlannerImpl(DPGreedyPlannerImpl &&) = default;
  virtual ~DPGreedyPlannerImpl() = default;
};

std::tuple<flow::Sequence, memory::Index, nlohmann::json>
PlannerImpl::Run(const flow::Flow &flow) {
  auto [sequence, json] = transform(flow);
  memory::Index index = run(sequence);
  return {std::move(sequence), std::move(index), json};
}

std::unique_ptr<Planner>
Planner::MakePlainLinearPlanner(context::Context &&context) {
  return std::make_unique<PlainLinearPlannerImpl>(std::move(context));
}

std::unique_ptr<Planner>
Planner::MakePlainGreedyPlanner(context::Context &&context) {
  return std::make_unique<PlainGreedyPlannerImpl>(std::move(context));
}

std::unique_ptr<Planner>
Planner::MakeDPGreedyPlanner(context::Context &&context) {
  return std::make_unique<DPGreedyPlannerImpl>(std::move(context));
}

PlannerImpl::PlannerImpl(context::Context &&context)
    : context_(std::move(context)) {}

memory::Index PlannerImpl::run(const flow::Sequence &sequence) {
  const std::vector<std::shared_ptr<flow::Node>> &nodes = sequence.GetNodes();
  const std::vector<std::shared_ptr<flow::Edge>> &edges = sequence.GetEdges();
  const std::vector<std::shared_ptr<flow::Region>> &regions =
      sequence.GetRegions();
  std::unordered_map<std::string, size_t> node_indices;
  size_t nodes_size = nodes.size();
  for (size_t i = 0; i < nodes_size; ++i) {
    const std::shared_ptr<flow::Node> &node = nodes[i];
    const std::string &name = node->GetName();
    node_indices.insert({name, i});
  }
  std::unordered_multimap<std::string, std::string> edge_refs;
  for (std::shared_ptr<flow::Edge> edge : edges) {
    if (std::shared_ptr<flow::MemoryEdge> memory_edge =
            std::dynamic_pointer_cast<flow::MemoryEdge>(edge)) {
      const std::string &name = memory_edge->GetName();
      std::shared_ptr<flow::Node> from = memory_edge->GetFrom(),
                                  to = memory_edge->GetTo();
      const std::string &from_name = from->GetName(), &to_name = to->GetName();
      edge_refs.insert({name, from_name});
      edge_refs.insert({name, to_name});
    }
  }
  std::unique_ptr<memory::Infos> infos = makeInfos(sequence);
  for (const std::shared_ptr<flow::Region> &region : regions) {
    if (region->NeedMemoryAllocation()) {
      const Meta &meta = region->GetMeta();
      size_t size = meta.GetSize(), min_index = -1, max_index = 0;
      std::string name = region->GetName();
      auto range = edge_refs.equal_range(name);
      for (auto it = range.first; it != range.second; ++it) {
        const std::string &ref = it->second;
        size_t index = node_indices.at(ref);
        min_index = std::min(min_index, index);
        max_index = std::max(max_index, index);
      }
#ifdef DEBUG
      assert(min_index != -1);
      assert(max_index != 0);
#endif
      memory::Info info(std::move(name), min_index, max_index, size);
      infos->Push(std::move(info));
    }
  }
  for (const std::shared_ptr<flow::Node> &node : nodes) {
    const size_t buffer_size = node->GetBufferSize();
    if (buffer_size > 0) {
      std::string name = node->GetName();
      size_t index = node_indices.at(name);
      memory::Info info(std::move(name), index, index, buffer_size);
      infos->Push(std::move(info));
    }
  }
  std::unique_ptr<memory::Plan> plan = makePlan();
  memory::Index index = plan->Run(*infos);
  return index;
}

std::tuple<flow::Sequence, nlohmann::json>
PlainPlannerImpl::transform(const flow::Flow &flow) {
  return {TopologicalSort(flow), nlohmann::json()};
}

std::tuple<flow::Sequence, nlohmann::json>
DynamicProgrammingPlannerImpl::transform(const flow::Flow &flow) {
  flow::Sequence sequence = TopologicalSort(flow);
  context::Context context = context_;
  std::shared_ptr<evaluation::DynamicProgrammingTable> dp_table =
      context_.MakeDynamicProgrammingTable();
  evaluation::DynamicProgrammingPlan plan = dp_table->Run(flow);
  std::vector<std::shared_ptr<flow::Region>> regions = sequence.GetRegions();
  for (std::shared_ptr<flow::Region> region : regions) {
    const std::string &name = region->GetName();
    if (plan.HasLayout(name)) {
      std::vector<size_t> layout = plan.GetLayout(name);
      region->SetLayout(std::move(layout));
    }
  }
  nlohmann::json json = {
      {"dp_table", dp_table->ToJson()},
      {"plan", plan.ToJson()},
  };
  return {std::move(sequence), std::move(json)};
}

std::unique_ptr<memory::Infos>
LinearPlannerImpl::makeInfos(const flow::Sequence &sequence) {
  return std::make_unique<memory::PlainInfos>();
}

std::unique_ptr<memory::Plan> LinearPlannerImpl::makePlan() {
  return std::make_unique<memory::LinearPlan>();
}

std::unique_ptr<memory::Infos>
GreedyPlannerImpl::makeInfos(const flow::Sequence &sequence) {
  return std::make_unique<memory::GreedyInfos>();
}

std::unique_ptr<memory::Plan> GreedyPlannerImpl::makePlan() {
  return std::make_unique<memory::GreedyPlan>();
}

PlainLinearPlannerImpl::PlainLinearPlannerImpl(context::Context &&context)
    : PlannerImpl(std::move(context)), LinearPlannerImpl(std::move(context)),
      PlainPlannerImpl(std::move(context)) {}

PlainGreedyPlannerImpl::PlainGreedyPlannerImpl(context::Context &&context)
    : PlannerImpl(std::move(context)), GreedyPlannerImpl(std::move(context)),
      PlainPlannerImpl(std::move(context)) {}

DPGreedyPlannerImpl::DPGreedyPlannerImpl(context::Context &&context)
    : PlannerImpl(std::move(context)), GreedyPlannerImpl(std::move(context)),
      DynamicProgrammingPlannerImpl(std::move(context)) {}

} // namespace worker
} // namespace cpu_transformers
