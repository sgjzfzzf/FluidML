#include "worker/planner.h"
#include "structure/flow/edge.h"
#include "structure/flow/node.h"
#include "structure/flow/region.h"
#include "structure/memory/greedy.h"
#include "structure/memory/index.h"
#include "structure/memory/info.h"
#include "structure/memory/linear.h"
#include <memory>
#include <unordered_map>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace worker {
Planner::Planner(std::unique_ptr<memory::Plan> &&plan)
    : plan_(std::move(plan)) {}

flow::Sequence Planner::FlowToSequence(const flow::Flow &flow) const {
  // Run the topological sort algorithm to get the sequence.
  flow::Sequence sequence;
  std::unordered_map<std::string, std::shared_ptr<flow::Node>> unvisited_nodes;
  std::list<std::shared_ptr<flow::Node>> waiting_nodes;
  const std::vector<std::shared_ptr<flow::Node>> &nodes = flow.GetNodes();
  const std::vector<std::shared_ptr<flow::Edge>> &edges = flow.GetEdges();
  for (const std::shared_ptr<flow::Node> &node : nodes) {
    unvisited_nodes.insert({node->GetName(), node});
  }
  for (std::shared_ptr<flow::Edge> edge : edges) {
    if (std::shared_ptr<flow::InputEdge> input_edge =
            std::dynamic_pointer_cast<flow::InputEdge>(edge)) {
      const std::string &to = input_edge->GetTo();
      auto it = unvisited_nodes.find(to);
      if (it != unvisited_nodes.end()) {
        waiting_nodes.push_back(std::move(it->second));
        unvisited_nodes.erase(it);
      }
    }
  }
  std::unordered_multimap<std::string, std::string> prev_nodes;
  for (std::shared_ptr<flow::Edge> edge : edges) {
    if (std::shared_ptr<flow::MemoryEdge> memory_edge =
            std::dynamic_pointer_cast<flow::MemoryEdge>(edge)) {
      std::string from = memory_edge->GetFrom();
      std::string to = memory_edge->GetTo();
      prev_nodes.insert({std::move(to), std::move(from)});
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

memory::Index Planner::Run(const flow::Sequence &sequence) const {
  const std::vector<std::shared_ptr<flow::Node>> &nodes = sequence.GetNodes();
  const std::vector<std::shared_ptr<flow::Edge>> &edges = sequence.GetEdges();
  const std::vector<std::shared_ptr<flow::Region>> &regions =
      sequence.GetRegions();
  std::unordered_map<std::string, size_t> node_indices;
  size_t nodesSize = nodes.size();
  for (size_t i = 0; i < nodesSize; ++i) {
    const std::shared_ptr<flow::Node> &node = nodes[i];
    const std::string &name = node->GetName();
    node_indices.insert({name, i});
  }
  std::unordered_multimap<std::string, std::string> edge_refs;
  for (std::shared_ptr<flow::Edge> edge : edges) {
    if (std::shared_ptr<flow::MemoryEdge> memory_edge =
            std::dynamic_pointer_cast<flow::MemoryEdge>(edge)) {
      const std::string &name = memory_edge->GetName();
      const std::string &from = memory_edge->GetFrom();
      const std::string &to = memory_edge->GetTo();
      edge_refs.insert({name, from});
      edge_refs.insert({name, to});
    }
  }
  std::unique_ptr<memory::Infos> infos = createInfos();
  for (const std::shared_ptr<flow::Region> &region : regions) {
    if (region->NeedMemoryAllocation()) {
      const Meta &meta = region->GetMeta();
      size_t size = meta.GetSize();
      std::string name = region->GetName();
      auto range = edge_refs.equal_range(name);
      size_t min_index = -1;
      size_t max_index = 0;
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
  memory::Index index = plan_->Run(*infos);
  return index;
}

LinearPlanner::LinearPlanner()
    : Planner(std::make_unique<memory::LinearPlan>()) {}

std::unique_ptr<memory::Infos> LinearPlanner::createInfos() const {
  return std::make_unique<memory::PlainInfos>();
}

GreedyPlanner::GreedyPlanner()
    : Planner(std::make_unique<memory::GreedyPlan>()) {}

std::unique_ptr<memory::Infos> GreedyPlanner::createInfos() const {
  return std::make_unique<memory::GreedyInfos>();
}

} // namespace worker
} // namespace cpu_transformers
