#include "evaluation/dp.h"
#include "evaluation/eval.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include "structure/context/factory.h"
#include "structure/flow/edge.h"
#include "structure/flow/flow.h"
#include "structure/flow/node.h"
#include "structure/flow/region.h"
#include "structure/kernel/kernel/kernel.h"
#include "utils/hash.h"
#include "utils/isa.hpp"
#include "utils/utils.h"
#include "worker/evaluator.h"
#include <limits>
#include <list>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace evaluation {

class SubFlowsBuilder {
public:
  struct Information {
    size_t distance;
    std::shared_ptr<flow::Edge> prev;
  };
  SubFlowsBuilder() = default;
  SubFlowsBuilder(const SubFlowsBuilder &dp) = delete;
  SubFlowsBuilder(SubFlowsBuilder &&dp) = default;
  std::vector<flow::Flow> Run(const flow::Flow &flow);
  std::vector<flow::Flow>
  Run(const flow::Flow &flow,
      std::unordered_map<std::string, Information> &djikstra_table);

private:
  size_t findLongestPathTo(
      const flow::Flow &flow,
      std::unordered_map<std::string, Information> &djikstra_table,
      std::shared_ptr<flow::Edge> edge);
};

class DPOnNoOverlapFlowWoker {
public:
  class KeyEqual;
  class KeyHash;

  class Key {
  public:
    Key(const std::shared_ptr<flow::Edge> &edge,
        const std::vector<size_t> &layout);
    Key(std::shared_ptr<flow::Edge> &&edge, std::vector<size_t> &&layout);
    Key(const Key &key) = default;
    Key(Key &&key) = default;
    Key &operator=(const Key &key) = default;
    Key &operator=(Key &&key) = default;
    friend class DPOnNoOverlapFlowWoker;
    friend struct KeyEqual;
    friend struct KeyHash;

  private:
    std::shared_ptr<flow::Edge> edge;
    std::vector<size_t> layout;
  };

  class Value {
  public:
    Value(size_t distance, const std::vector<Key> &prevs);
    Value(size_t distance, std::vector<Key> &&prevs);
    Value(const Value &value) = default;
    Value(Value &&value) = default;
    Value &operator=(const Value &value) = default;
    Value &operator=(Value &&value) = default;
    friend class DPOnNoOverlapFlowWoker;

  private:
    size_t distance;
    std::vector<Key> prevs;
  };

  struct KeyEqual {
    bool operator()(const Key &lhs, const Key &rhs) const;
  };

  struct KeyHash {
    size_t operator()(const Key &edge) const;
  };

  DPOnNoOverlapFlowWoker(std::shared_ptr<worker::Evaluator> &&evaluator);
  DPOnNoOverlapFlowWoker(const DPOnNoOverlapFlowWoker &runner) = delete;
  DPOnNoOverlapFlowWoker(DPOnNoOverlapFlowWoker &&runner) = default;
  DynamicProgrammingPlan Run(const flow::Flow &flow);
  DynamicProgrammingPlan
  Run(const flow::Flow &flow,
      std::unordered_map<Key, Value, KeyHash, KeyEqual> &dp_table);

private:
  size_t runOn(const flow::Flow &flow,
               std::unordered_map<Key, Value, KeyHash, KeyEqual> &dp_table,
               std::shared_ptr<flow::Edge> edge,
               const std::vector<size_t> &layout);
  std::shared_ptr<worker::Evaluator> evaluator_;
};

class DynamicProgrammingTableImpl : public DynamicProgrammingTable {
public:
  DynamicProgrammingTableImpl(context::Context &&context);
  DynamicProgrammingTableImpl(const DynamicProgrammingTableImpl &table) =
      delete;
  DynamicProgrammingTableImpl(DynamicProgrammingTableImpl &&table) = default;
  virtual ~DynamicProgrammingTableImpl() = default;
  DynamicProgrammingPlan Run(const flow::Flow &flow) override;

private:
  std::shared_ptr<worker::Evaluator> getEvaluator(const flow::Flow &flow);
  context::Context context_;
  std::shared_ptr<worker::Evaluator> evaluator_;
};

std::shared_ptr<DynamicProgrammingTable>
DynamicProgrammingTable::Make(context::Context &&context) {
  return std::make_shared<DynamicProgrammingTableImpl>(std::move(context));
}

std::vector<flow::Flow> SubFlowsBuilder::Run(const flow::Flow &flow) {
  std::unordered_map<std::string, Information> djikstra_table;
  return Run(flow, djikstra_table);
}

std::vector<flow::Flow> SubFlowsBuilder::Run(
    const flow::Flow &flow,
    std::unordered_map<std::string, Information> &djikstra_table) {
  std::vector<std::shared_ptr<flow::Edge>> edges = flow.GetEdges();
  std::vector<std::shared_ptr<flow::Node>> nodes = flow.GetNodes();
  size_t farthest_distance = 0;
  std::shared_ptr<flow::Edge> farthest_edge = nullptr;
  std::vector<flow::Flow> subflows;
  for (const std::shared_ptr<flow::Edge> &edge : edges) {
#ifdef DEBUG
    assert(edge != nullptr);
#endif
    size_t distance = findLongestPathTo(flow, djikstra_table, edge);
    if (distance > farthest_distance) {
      farthest_distance = distance;
      farthest_edge = edge;
    }
  }
#ifdef DEBUG
  assert(farthest_distance != 0);
  assert(farthest_edge != nullptr);
#endif
  std::unordered_set<std::shared_ptr<flow::Edge>> path;
  for (std::shared_ptr<flow::Edge> edge = farthest_edge; edge != nullptr;
       edge = djikstra_table[edge->GetName()].prev) {
    path.insert(edge);
  }
  flow::Flow main_subflow;
  std::unordered_set<std::shared_ptr<flow::Node>> main_subflow_nodes;
  std::unordered_set<std::shared_ptr<flow::Edge>> main_subflow_edges;
  // Check every edge on the whole path.
  for (std::shared_ptr<flow::Edge> edge : path) {
    // If the edge is an input edge, we should check its corresponding node.
    if (std::shared_ptr<flow::OwnToEdge> own_to_edge =
            std::dynamic_pointer_cast<flow::OwnToEdge>(edge)) {
      std::shared_ptr<flow::Node> to = own_to_edge->GetTo();
#ifdef DEBUG
      assert(isa<flow::SingleInputNode>(to) || isa<flow::DoubleInputsNode>(to));
#endif
      if (std::shared_ptr<flow::DoubleInputsNode> double_inputs_node =
              std::dynamic_pointer_cast<flow::DoubleInputsNode>(to)) {
        // If the node is a double inputs node, we should check which edge
        // doesn't
        // exist in the path. There are two things to do.
        //
        // First, create a new input edge as a substitute, so we can create a
        // whole path even lacking them.
        //
        // Second, create a new node as a substitute, because the original node
        // has a missing edge which is not in the path. In this way, we protect
        // the integrity of the path.
        const std::string &lhs_name = double_inputs_node->GetLhsAsString(),
                          &rhs_name = double_inputs_node->GetRhsAsString();
        std::vector<std::shared_ptr<flow::Edge>> lhs_edges =
                                                     flow.GetEdges(lhs_name),
                                                 rhs_edges =
                                                     flow.GetEdges(rhs_name);
        std::shared_ptr<flow::Edge> lhs_edge =
                                        flow.GetLhsEdge(*double_inputs_node),
                                    rhs_edge =
                                        flow.GetRhsEdge(*double_inputs_node),
                                    extra_edge = nullptr;
#ifdef DEBUG
        assert(lhs_edge != nullptr);
        assert(rhs_edge != nullptr);
#endif
        if (lhs_edge == edge) {
          extra_edge = std::move(rhs_edge);
        } else if (rhs_edge == edge) {
          extra_edge = std::move(lhs_edge);
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
        std::shared_ptr<flow::InputEdge> extra_input_edge =
            std::make_shared<flow::InputEdge>(extra_edge->GetRegion(),
                                              std::move(double_inputs_node));
        main_subflow_edges.insert(std::move(extra_input_edge));
      }
      main_subflow_nodes.insert(std::move(to));
    }
    main_subflow_edges.insert(std::move(edge));
  }
  for (std::shared_ptr<flow::Node> node : main_subflow_nodes) {
    main_subflow.PutNode(std::move(node));
  }
  for (std::shared_ptr<flow::Edge> edge : main_subflow_edges) {
    std::shared_ptr<flow::Region> region = edge->GetRegion();
    if (main_subflow.GetRegion(region->GetName()) == nullptr) {
      main_subflow.PutRegion(std::move(region));
    }
    main_subflow.PutEdge(std::move(edge));
  }
#ifdef DEBUG
  assert(main_subflow.Check());
  assert(main_subflow.IsNoOverlapFlow());
#endif
  subflows.push_back(std::move(main_subflow));
  // Split the remained nodes and edges into several directed acyclic graphs,
  // called branches.
  std::unordered_set<std::shared_ptr<flow::Node>> remained_nodes;
  std::vector<std::unordered_set<std::shared_ptr<flow::Edge>>> branches_edges;
  std::vector<std::unordered_set<std::shared_ptr<flow::Node>>> branches_nodes;
  for (std::shared_ptr<flow::Node> node : nodes) {
    if (main_subflow_nodes.find(node) == main_subflow_nodes.end()) {
      // If the node is not in the main subflow, it belongs to some branch. So
      // store them in a list for further processing.
      remained_nodes.insert(node);
    }
  }
  while (!remained_nodes.empty()) {
    auto first_it = remained_nodes.begin();
    std::shared_ptr<flow::Node> first = std::move(*first_it);
    remained_nodes.erase(first_it);
    std::list<std::shared_ptr<flow::Node>> directed_graph_nodes = {first};
    std::unordered_set<std::shared_ptr<flow::Node>> branch_nodes;
    while (!directed_graph_nodes.empty()) {
      std::shared_ptr<flow::Node> node =
          std::move(directed_graph_nodes.front());
      directed_graph_nodes.pop_front();
      if (std::shared_ptr<flow::SingleInputNode> single_input_node =
              std::dynamic_pointer_cast<flow::SingleInputNode>(node)) {
        std::shared_ptr<flow::Edge> input_edge =
                                        flow.GetInputEdge(*single_input_node),
                                    output_edge =
                                        flow.GetOutputEdge(*single_input_node);
#ifdef DEBUG
        assert(input_edge != nullptr);
        assert(output_edge != nullptr);
#endif
        if (std::shared_ptr<flow::OwnFromEdge> own_from_edge =
                std::dynamic_pointer_cast<flow::OwnFromEdge>(input_edge)) {
          std::shared_ptr<flow::Node> from = own_from_edge->GetFrom();
          if (main_subflow_nodes.find(from) == main_subflow_nodes.end() &&
              branch_nodes.find(from) == branch_nodes.end()) {
            directed_graph_nodes.push_back(std::move(from));
          }
        }
        if (std::shared_ptr<flow::OwnToEdge> own_to_edge =
                std::dynamic_pointer_cast<flow::OwnToEdge>(output_edge)) {
          std::shared_ptr<flow::Node> to = own_to_edge->GetTo();
          if (main_subflow_nodes.find(to) == main_subflow_nodes.end() &&
              branch_nodes.find(to) == branch_nodes.end()) {
            directed_graph_nodes.push_back(std::move(to));
          }
        }
      } else if (std::shared_ptr<flow::DoubleInputsNode> double_inputs_node =
                     std::dynamic_pointer_cast<flow::DoubleInputsNode>(node)) {
        std::shared_ptr<flow::Edge> lhs_edge =
                                        flow.GetLhsEdge(*double_inputs_node),
                                    rhs_edge =
                                        flow.GetRhsEdge(*double_inputs_node),
                                    output_edge =
                                        flow.GetOutputEdge(*double_inputs_node);
#ifdef DEBUG
        assert(lhs_edge != nullptr);
        assert(rhs_edge != nullptr);
        assert(output_edge != nullptr);
#endif
        if (std::shared_ptr<flow::OwnFromEdge> own_from_edge =
                std::dynamic_pointer_cast<flow::OwnFromEdge>(lhs_edge)) {
          std::shared_ptr<flow::Node> from = own_from_edge->GetFrom();
          if (main_subflow_nodes.find(from) == main_subflow_nodes.end() &&
              branch_nodes.find(from) == branch_nodes.end()) {
            directed_graph_nodes.push_back(std::move(from));
          }
        }
        if (std::shared_ptr<flow::OwnFromEdge> own_from_edge =
                std::dynamic_pointer_cast<flow::OwnFromEdge>(rhs_edge)) {
          std::shared_ptr<flow::Node> from = own_from_edge->GetFrom();
          if (main_subflow_nodes.find(from) == main_subflow_nodes.end() &&
              branch_nodes.find(from) == branch_nodes.end()) {
            directed_graph_nodes.push_back(std::move(from));
          }
        }
        if (std::shared_ptr<flow::OwnToEdge> own_to_edge =
                std::dynamic_pointer_cast<flow::OwnToEdge>(output_edge)) {
          std::shared_ptr<flow::Node> to = own_to_edge->GetTo();
          if (main_subflow_nodes.find(to) == main_subflow_nodes.end() &&
              branch_nodes.find(to) == branch_nodes.end()) {
            directed_graph_nodes.push_back(std::move(to));
          }
        }
      } else {
#ifdef DEBUG
        assert(false && "unreachable");
#else
        __builtin_unreachable();
#endif
      }
      branch_nodes.insert(std::move(node));
    }
    branches_nodes.push_back(std::move(branch_nodes));
  }
  for (const std::unordered_set<std::shared_ptr<flow::Node>> &branch_nodes :
       branches_nodes) {
    flow::Flow branch_subflow;
    for (std::shared_ptr<flow::Node> node : branch_nodes) {
      if (std::shared_ptr<flow::SingleInputNode> single_input_node =
              std::dynamic_pointer_cast<flow::SingleInputNode>(node)) {
        std::shared_ptr<flow::OwnToEdge> input_edge =
            flow.GetInputEdge(*single_input_node);
        std::shared_ptr<flow::OwnFromEdge> output_edge =
            flow.GetOutputEdge(*single_input_node);
#ifdef DEBUG
        assert(input_edge != nullptr);
        assert(output_edge != nullptr);
        assert(path.find(input_edge) == path.end());
        assert(path.find(output_edge) == path.end());
#endif
        std::shared_ptr<flow::OwnToEdge> real_input_edge = nullptr;
        std::shared_ptr<flow::OwnFromEdge> real_output_edge = nullptr;
        if (std::shared_ptr<flow::MemoryEdge> input_memory_edge =
                std::dynamic_pointer_cast<flow::MemoryEdge>(input_edge)) {
          std::shared_ptr<flow::Node> from = input_memory_edge->GetFrom();
          if (branch_nodes.find(from) != branch_nodes.end()) {
            real_input_edge = input_edge;
          } else {
            // If the input node of this edge isn't in this branch, we need to
            // modify it into a new `InputEdge` to keep the new subflow as a
            // whole flow.
            std::shared_ptr<flow::SingleInputNode> node_clone =
                single_input_node;
            real_input_edge = std::make_shared<flow::InputEdge>(
                input_edge->GetRegion(), std::move(node_clone));
          }
        } else if (isa<flow::InputEdge>(input_edge) ||
                   isa<flow::ConstantEdge>(input_edge)) {
          real_input_edge = input_edge;
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
        if (std::shared_ptr<flow::MemoryEdge> output_memory_edge =
                std::dynamic_pointer_cast<flow::MemoryEdge>(output_edge)) {
          std::shared_ptr<flow::Node> to = output_memory_edge->GetTo();
          if (branch_nodes.find(to) != branch_nodes.end()) {
            real_output_edge = output_edge;
          } else {
            std::shared_ptr<flow::SingleInputNode> node_clone =
                single_input_node;
            real_output_edge = std::make_shared<flow::OutputEdge>(
                output_edge->GetRegion(), std::move(node_clone));
          }
        } else if (std::shared_ptr<flow::OutputEdge> output_output_edge =
                       std::dynamic_pointer_cast<flow::OutputEdge>(
                           output_edge)) {
          real_output_edge = output_edge;
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
#ifdef DEBUG
        assert(real_input_edge != nullptr);
        assert(real_output_edge != nullptr);
#endif
        std::shared_ptr<flow::Region> input_region =
                                          real_input_edge->GetRegion(),
                                      output_region =
                                          real_output_edge->GetRegion();
        if (branch_subflow.GetRegion(input_region->GetName()) == nullptr) {
          branch_subflow.PutRegion(std::move(input_region));
        }
        if (branch_subflow.GetRegion(output_region->GetName()) == nullptr) {
          branch_subflow.PutRegion(std::move(output_region));
        }
        branch_subflow.PutNode(std::move(node));
        std::vector<std::shared_ptr<flow::Edge>>
            potential_input_edges =
                branch_subflow.GetEdges(real_input_edge->GetName()),
            potential_output_edges =
                branch_subflow.GetEdges(real_output_edge->GetName());
        if (std::find(potential_input_edges.begin(),
                      potential_input_edges.end(),
                      real_input_edge) == potential_input_edges.end()) {
          branch_subflow.PutEdge(std::move(real_input_edge));
        }
        if (std::find(potential_output_edges.begin(),
                      potential_output_edges.end(),
                      real_output_edge) == potential_output_edges.end()) {
          branch_subflow.PutEdge(std::move(real_output_edge));
        }
      } else if (std::shared_ptr<flow::DoubleInputsNode> double_inputs_node =
                     std::dynamic_pointer_cast<flow::DoubleInputsNode>(node)) {
        std::shared_ptr<flow::OwnToEdge> lhs_edge = flow.GetLhsEdge(
                                             *double_inputs_node),
                                         rhs_edge = flow.GetRhsEdge(
                                             *double_inputs_node);
        std::shared_ptr<flow::OwnFromEdge> output_edge =
            flow.GetOutputEdge(*double_inputs_node);
#ifdef DEBUG
        assert(lhs_edge != nullptr);
        assert(rhs_edge != nullptr);
        assert(output_edge != nullptr);
        assert(path.find(lhs_edge) == path.end());
        assert(path.find(rhs_edge) == path.end());
        assert(path.find(output_edge) == path.end());
#endif
        std::shared_ptr<flow::OwnToEdge> real_lhs_edge = nullptr,
                                         real_rhs_edge = nullptr;
        std::shared_ptr<flow::OwnFromEdge> real_output_edge = nullptr;
        if (std::shared_ptr<flow::MemoryEdge> lhs_memory_edge =
                std::dynamic_pointer_cast<flow::MemoryEdge>(lhs_edge)) {
          std::shared_ptr<flow::Node> from = lhs_memory_edge->GetFrom();
          if (branch_nodes.find(from) != branch_nodes.end()) {
            real_lhs_edge = lhs_edge;
          } else {
            std::shared_ptr<flow::DoubleInputsNode> node_clone =
                double_inputs_node;
            real_lhs_edge = std::make_shared<flow::InputEdge>(
                lhs_edge->GetRegion(), std::move(node_clone));
          }
        } else if (isa<flow::OwnToEdge>(lhs_edge) ||
                   isa<flow::InputEdge>(lhs_edge)) {
          real_lhs_edge = lhs_edge;
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
        if (std::shared_ptr<flow::MemoryEdge> rhs_memory_edge =
                std::dynamic_pointer_cast<flow::MemoryEdge>(rhs_edge)) {
          std::shared_ptr<flow::Node> from = rhs_memory_edge->GetFrom();
          if (branch_nodes.find(from) != branch_nodes.end()) {
            real_rhs_edge = rhs_edge;
          } else {
            std::shared_ptr<flow::DoubleInputsNode> node_clone =
                double_inputs_node;
            real_rhs_edge = std::make_shared<flow::InputEdge>(
                rhs_edge->GetRegion(), std::move(node_clone));
          }
        } else if (isa<flow::InputEdge>(rhs_edge) ||
                   isa<flow::ConstantEdge>(rhs_edge)) {
          real_rhs_edge = rhs_edge;
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
        if (std::shared_ptr<flow::MemoryEdge> output_memory_edge =
                std::dynamic_pointer_cast<flow::MemoryEdge>(output_edge)) {
          std::shared_ptr<flow::Node> to = output_memory_edge->GetTo();
          if (branch_nodes.find(to) != branch_nodes.end()) {
            real_output_edge = output_edge;
          } else {
            std::shared_ptr<flow::DoubleInputsNode> node_clone =
                double_inputs_node;
            real_output_edge = std::make_shared<flow::OutputEdge>(
                output_edge->GetRegion(), std::move(node_clone));
          }
        } else if (std::shared_ptr<flow::OutputEdge> output_output_edge =
                       std::dynamic_pointer_cast<flow::OutputEdge>(
                           output_edge)) {
          real_output_edge = output_edge;
        } else {
#ifdef DEBUG
          assert(false && "unreachable");
#else
          __builtin_unreachable();
#endif
        }
#ifdef DEBUG
        assert(real_lhs_edge != nullptr);
        assert(real_rhs_edge != nullptr);
        assert(real_output_edge != nullptr);
#endif
        std::shared_ptr<flow::Region> lhs_region = real_lhs_edge->GetRegion(),
                                      rhs_region = real_rhs_edge->GetRegion(),
                                      output_region =
                                          real_output_edge->GetRegion();
        branch_subflow.PutNode(std::move(node));
        std::vector<std::shared_ptr<flow::Edge>>
            potential_lhs_edges =
                branch_subflow.GetEdges(real_lhs_edge->GetName()),
            potential_rhs_edges =
                branch_subflow.GetEdges(real_rhs_edge->GetName()),
            potential_output_edges =
                branch_subflow.GetEdges(real_output_edge->GetName());
        if (std::find(potential_lhs_edges.begin(), potential_lhs_edges.end(),
                      real_lhs_edge) == potential_lhs_edges.end()) {
          branch_subflow.PutEdge(std::move(real_lhs_edge));
        }
        if (std::find(potential_rhs_edges.begin(), potential_rhs_edges.end(),
                      real_rhs_edge) == potential_rhs_edges.end()) {
          branch_subflow.PutEdge(std::move(real_rhs_edge));
        }
        if (std::find(potential_output_edges.begin(),
                      potential_output_edges.end(),
                      real_output_edge) == potential_output_edges.end()) {
          branch_subflow.PutEdge(std::move(real_output_edge));
        }
        if (branch_subflow.GetRegion(lhs_region->GetName()) == nullptr) {
          branch_subflow.PutRegion(std::move(lhs_region));
        }
        if (branch_subflow.GetRegion(rhs_region->GetName()) == nullptr) {
          branch_subflow.PutRegion(std::move(rhs_region));
        }
        if (branch_subflow.GetRegion(output_region->GetName()) == nullptr) {
          branch_subflow.PutRegion(std::move(output_region));
        }
      }
    }
#ifdef DEBUG
    assert(branch_subflow.Check());
#endif
    SubFlowsBuilder builder;
    std::vector<flow::Flow> flows = builder.Run(branch_subflow);
    for (flow::Flow &flow : flows) {
      subflows.push_back(std::move(flow));
    }
  }
#ifdef DEBUG
  for (const flow::Flow &flow : subflows) {
    assert(flow.Check());
    assert(flow.IsNoOverlapFlow());
  }
#endif
  return subflows;
}

size_t SubFlowsBuilder::findLongestPathTo(
    const flow::Flow &flow,
    std::unordered_map<std::string, Information> &djikstra_table,
    std::shared_ptr<flow::Edge> edge) {
  auto it = djikstra_table.find(edge->GetName());
  if (it != djikstra_table.end()) {
    auto [distance, _] = it->second;
    return distance;
  }
  size_t distance = -1;
  std::shared_ptr<flow::Edge> prev = nullptr;
  if (isa<flow::InputEdge>(edge) || isa<flow::ConstantEdge>(edge)) {
    distance = 0;
  } else if (std::shared_ptr<flow::OwnFromEdge> own_from_edge =
                 std::dynamic_pointer_cast<flow::OwnFromEdge>(edge)) {
    std::shared_ptr<flow::Node> from = own_from_edge->GetFrom();
    if (std::shared_ptr<flow::SingleInputNode> single_input_node =
            std::dynamic_pointer_cast<flow::SingleInputNode>(from)) {
      const std::string &input_name = single_input_node->GetInputAsString();
      std::vector<std::shared_ptr<flow::Edge>> edges =
          flow.GetEdges(input_name);
      std::shared_ptr<flow::Edge> input = nullptr;
      for (std::shared_ptr<flow::Edge> edge : edges) {
        if (std::shared_ptr<flow::OwnToEdge> own_to_edge =
                std::dynamic_pointer_cast<flow::OwnToEdge>(edge)) {
          if (own_to_edge->GetTo() == single_input_node) {
            input = std::move(edge);
            break;
          }
        }
      }
#ifdef DEBUG
      assert(input != nullptr);
#endif
      // TODO: for a more precise result, replace `+1` with our estimation
      distance = findLongestPathTo(flow, djikstra_table, input) + 1;
      prev = std::move(input);
    } else if (std::shared_ptr<flow::DoubleInputsNode> double_inputs_node =
                   std::dynamic_pointer_cast<flow::DoubleInputsNode>(from)) {
      const std::string &lhs_name = double_inputs_node->GetLhsAsString(),
                        &rhs_name = double_inputs_node->GetRhsAsString();
      const std::vector<std::shared_ptr<flow::Edge>> &lhs_edges = flow.GetEdges(
                                                         lhs_name),
                                                     &rhs_edges = flow.GetEdges(
                                                         rhs_name);
      std::shared_ptr<flow::Edge> lhs = nullptr, rhs = nullptr;
      for (std::shared_ptr<flow::Edge> edge : lhs_edges) {
        if (std::shared_ptr<flow::OwnToEdge> own_to_edge =
                std::dynamic_pointer_cast<flow::OwnToEdge>(edge)) {
          if (own_to_edge->GetTo() == double_inputs_node) {
            lhs = std::move(edge);
            break;
          }
        }
      }
      for (std::shared_ptr<flow::Edge> edge : rhs_edges) {
        if (std::shared_ptr<flow::OwnToEdge> own_to_edge =
                std::dynamic_pointer_cast<flow::OwnToEdge>(edge)) {
          if (own_to_edge->GetTo() == double_inputs_node) {
            rhs = std::move(edge);
            break;
          }
        }
      }
#ifdef DEBUG
      assert(lhs != nullptr);
      assert(rhs != nullptr);
#endif
      const size_t lhs_distance = findLongestPathTo(flow, djikstra_table, lhs),
                   rhs_distance = findLongestPathTo(flow, djikstra_table, rhs);
      if (lhs_distance > rhs_distance) {
        distance = lhs_distance + 1;
        prev = std::move(lhs);
      } else {
        distance = rhs_distance + 1;
        prev = std::move(rhs);
      }
#ifdef DEBUG
      assert(prev != nullptr);
#endif
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
#ifdef DEBUG
  assert(distance != -1);
#endif
  djikstra_table.insert({edge->GetName(), {distance, prev}});
  return distance;
}

DPOnNoOverlapFlowWoker::Key::Key(const std::shared_ptr<flow::Edge> &edge,
                                 const std::vector<size_t> &layout)
    : edge(edge), layout(layout) {}

DPOnNoOverlapFlowWoker::Key::Key(std::shared_ptr<flow::Edge> &&edge,
                                 std::vector<size_t> &&layout)
    : edge(std::move(edge)), layout(std::move(layout)) {}

DPOnNoOverlapFlowWoker::Value::Value(size_t distance,
                                     const std::vector<Key> &prevs)
    : distance(distance), prevs(prevs) {}

DPOnNoOverlapFlowWoker::Value::Value(size_t distance, std::vector<Key> &&prevs)
    : distance(distance), prevs(std::move(prevs)) {}

DPOnNoOverlapFlowWoker::DPOnNoOverlapFlowWoker(
    std::shared_ptr<worker::Evaluator> &&evaluator)
    : evaluator_(std::move(evaluator)) {}

DynamicProgrammingPlan DPOnNoOverlapFlowWoker::Run(const flow::Flow &flow) {
  std::unordered_map<Key, Value, KeyHash, KeyEqual> dp_table;
  return Run(flow, dp_table);
}

DynamicProgrammingPlan DPOnNoOverlapFlowWoker::Run(
    const flow::Flow &flow,
    std::unordered_map<Key, Value, KeyHash, KeyEqual> &dp_table) {
  std::vector<std::shared_ptr<flow::Edge>> edges = flow.GetEdges();
  std::unordered_map<std::string, std::vector<size_t>> layout_table;
  for (std::shared_ptr<flow::Edge> edge : edges) {
    if (isa<flow::OutputEdge>(edge)) {
      const std::string &name = edge->GetName();
      const Meta &meta = edge->GetMeta();
      const std::vector<size_t> &layout = edge->GetLayout();
      runOn(flow, dp_table, edge, layout);
      std::list<std::tuple<std::shared_ptr<flow::Edge>, std::vector<size_t>>>
          queue = {{edge, layout}};
      layout_table.insert({name, layout});
      while (!queue.empty()) {
        auto [edge, layout] = std::move(queue.front());
        queue.pop_front();
        auto it = dp_table.find({edge, layout});
#ifdef DEBUG
        assert(it != dp_table.end());
#endif
        auto [_, deps] = it->second;
        for (const Key &dep : deps) {
#ifdef DEBUG
          assert(dep.edge != nullptr);
#endif
          const std::string &name = dep.edge->GetName();
#ifdef DEBUG
          auto cannot_exist_it = dp_table.find(dep);
          assert(cannot_exist_it != dp_table.end());
#endif
          layout_table.insert({name, dep.layout});
          queue.push_back({dep.edge, dep.layout});
        }
      }
    }
  }
  return DynamicProgrammingPlan(std::move(layout_table));
}

bool DPOnNoOverlapFlowWoker::KeyEqual::operator()(const Key &lhs,
                                                  const Key &rhs) const {
  return lhs.edge == rhs.edge && lhs.layout == rhs.layout;
}

size_t DPOnNoOverlapFlowWoker::KeyHash::operator()(const Key &edge) const {
  size_t hash = 0;
  std::hash<std::shared_ptr<flow::Edge>> edge_hash;
  std::hash<int64_t> layout_hash;
  hash ^= edge_hash(edge.edge);
  for (int64_t i : edge.layout) {
    hash ^= layout_hash(i) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  return hash;
}

size_t DPOnNoOverlapFlowWoker::runOn(
    const flow::Flow &flow,
    std::unordered_map<Key, Value, KeyHash, KeyEqual> &dp_table,
    std::shared_ptr<flow::Edge> edge, const std::vector<size_t> &layout) {
  auto it = dp_table.find({edge, layout});
  if (it != dp_table.end()) {
    return it->second.distance;
  }
  if (isa<flow::InputEdge>(edge) || isa<flow::ConstantEdge>(edge)) {
    constexpr size_t kInputEdgeTimeCost = 0;
    Value result = {kInputEdgeTimeCost, {}};
    dp_table.insert({{edge, layout}, result});
    return kInputEdgeTimeCost;
  } else if (std::shared_ptr<flow::OwnFromEdge> own_from_edge =
                 std::dynamic_pointer_cast<flow::OwnFromEdge>(edge)) {
    std::shared_ptr<flow::Node> node = own_from_edge->GetFrom();
    const std::string &node_name = node->GetName();
    if (std::shared_ptr<flow::SingleInputNode> single_input_node =
            std::dynamic_pointer_cast<flow::SingleInputNode>(node)) {
      std::shared_ptr<flow::OwnToEdge> input_edge =
          flow.GetInputEdge(*single_input_node);
      const Meta &meta = input_edge->GetMeta();
      const std::vector<int64_t> &shape = meta.GetShape();
      const size_t shape_len = shape.size();
      std::vector<std::vector<size_t>> input_layouts =
          utils::GenAllOrders(shape_len);
      size_t min_time_cost = std::numeric_limits<size_t>::max();
      std::vector<Key> deps;
      for (const std::vector<size_t> &input_layout : input_layouts) {
        evaluation::SingleInputKernelEval &eval =
            evaluator_->GetSingleInputEval(node_name);
        const size_t time_cost =
            eval.GetTimeCost(input_layout, layout) +
            runOn(flow, dp_table, input_edge, input_layout);
        if (min_time_cost > time_cost) {
          min_time_cost = time_cost;
          deps = {{input_edge, input_layout}};
        }
      }
      dp_table.insert({{edge, layout}, {min_time_cost, std::move(deps)}});
      return min_time_cost;
    } else if (std::shared_ptr<flow::DoubleInputsNode> double_inputs_node =
                   std::dynamic_pointer_cast<flow::DoubleInputsNode>(node)) {
      std::shared_ptr<flow::OwnToEdge> lhs_edge =
                                           flow.GetLhsEdge(*double_inputs_node),
                                       rhs_edge =
                                           flow.GetRhsEdge(*double_inputs_node);
      const Meta &lhs_meta = lhs_edge->GetMeta(),
                 &rhs_meta = rhs_edge->GetMeta();
      const std::vector<int64_t> &lhs_shape = lhs_meta.GetShape(),
                                 &rhs_shape = rhs_meta.GetShape();
      const size_t lhs_shape_len = lhs_shape.size(),
                   rhs_shape_len = rhs_shape.size();
      std::vector<std::vector<size_t>> lhs_layouts =
                                           utils::GenAllOrders(lhs_shape_len),
                                       rhs_layouts =
                                           utils::GenAllOrders(rhs_shape_len);
      size_t min_time_cost = std::numeric_limits<size_t>::max();
      std::vector<Key> deps;
      for (const std::vector<size_t> &lhs_layout : lhs_layouts) {
        for (const std::vector<size_t> &rhs_layout : rhs_layouts) {
          evaluation::DoubleInputsKernelEval &eval =
              evaluator_->GetDoubleInputsEval(node_name);
          const size_t time_cost =
              eval.GetTimeCost(lhs_layout, rhs_layout, layout) +
              runOn(flow, dp_table, lhs_edge, lhs_layout) +
              runOn(flow, dp_table, rhs_edge, rhs_layout);
          if (min_time_cost > time_cost) {
            min_time_cost = time_cost;
            deps = {{lhs_edge, lhs_layout}, {rhs_edge, rhs_layout}};
          }
        }
      }
      dp_table.insert({{edge, layout}, {min_time_cost, std::move(deps)}});
      return min_time_cost;
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
}

DynamicProgrammingPlan::DynamicProgrammingPlan(
    std::unordered_map<std::string, std::vector<size_t>> &&plan)
    : plan_(std::move(plan)) {}

bool DynamicProgrammingPlan::HasLayout(const std::string &name) const {
  return plan_.find(name) != plan_.end();
}

const std::vector<size_t> &
DynamicProgrammingPlan::GetLayout(const std::string &name) const {
  auto it = plan_.find(name);
#ifdef DEBUG
  assert(it != plan_.end());
#endif
  return it->second;
}

DynamicProgrammingPlan Merge(const DynamicProgrammingPlan &lhs,
                             const DynamicProgrammingPlan &rhs) {
  std::unordered_map<std::string, std::vector<size_t>> plan;
  const std::unordered_map<std::string, std::vector<size_t>> &lhs_plan =
                                                                 lhs.plan_,
                                                             &rhs_plan =
                                                                 rhs.plan_;
  for (const auto &[name, layout] : lhs_plan) {
    plan.insert({name, layout});
  }
  for (const auto &[name, layout] : rhs_plan) {
    plan.insert({name, layout});
  }
  return DynamicProgrammingPlan(std::move(plan));
}

nlohmann::json DynamicProgrammingPlan::ToJson() const {
  return nlohmann::json(plan_);
}

std::ostream &operator<<(std::ostream &os, const DynamicProgrammingPlan &plan) {
  os << plan.ToJson();
  return os;
}

DynamicProgrammingTableImpl::DynamicProgrammingTableImpl(
    context::Context &&context)
    : context_(std::move(context)) {}

DynamicProgrammingPlan
DynamicProgrammingTableImpl::Run(const flow::Flow &flow) {
  SubFlowsBuilder builder;
  std::vector<flow::Flow> subflows = builder.Run(flow);
  DynamicProgrammingPlan plan;
  for (const flow::Flow &subflow : subflows) {
#ifdef DEBUG
    assert(subflow.IsNoOverlapFlow());
#endif
    std::shared_ptr<worker::Evaluator> evaluator = getEvaluator(flow);
    DPOnNoOverlapFlowWoker worker(std::move(evaluator));
    plan = Merge(plan, worker.Run(subflow));
  }
  return plan;
}

std::shared_ptr<worker::Evaluator>
DynamicProgrammingTableImpl::getEvaluator(const flow::Flow &flow) {
  if (evaluator_ != nullptr) {
    return evaluator_;
  }
  evaluator_ = worker::Evaluator::Make();
  context::Factory &factory = context_->GetFactory();
  const std::vector<std::shared_ptr<flow::Node>> &nodes = flow.GetNodes();
  for (std::shared_ptr<flow::Node> node : nodes) {
    std::shared_ptr<kernel::KernelGenerator> kgenerator =
        factory.MakeKernelGenerator(*node);
    size_t time_cost = 0;
    if (std::shared_ptr<kernel::SingleInputWithoutBufferKernelGenerator>
            generator = std::dynamic_pointer_cast<
                kernel::SingleInputWithoutBufferKernelGenerator>(kgenerator)) {
      std::shared_ptr<flow::SingleInputWithoutBufferNode> ptr =
          std::dynamic_pointer_cast<flow::SingleInputWithoutBufferNode>(node);
#ifdef DEBUG
      assert(ptr != nullptr);
#endif
      std::shared_ptr<flow::Region> input = ptr->GetInput();
      std::shared_ptr<flow::Region> output = ptr->GetOutput();
#ifdef DEBUG
      assert(input != nullptr);
      assert(output != nullptr);
#endif
      std::shared_ptr<evaluation::SingleInputWithoutBufferKernelEval> eval =
          std::make_shared<evaluation::SingleInputWithoutBufferKernelEval>(
              std::move(generator));
      std::string name = ptr->GetName();
      evaluator_->RegisterEval(std::move(name), std::move(eval));
    } else if (std::shared_ptr<kernel::SingleInputWithBufferKernelGenerator>
                   generator = std::dynamic_pointer_cast<
                       kernel::SingleInputWithBufferKernelGenerator>(
                       kgenerator)) {
      std::shared_ptr<flow::SingleInputWithBufferNode> ptr =
          std::dynamic_pointer_cast<flow::SingleInputWithBufferNode>(node);
#ifdef DEBUG
      assert(ptr != nullptr);
#endif
      std::shared_ptr<flow::Region> input = ptr->GetInput();
      std::shared_ptr<flow::Region> output = ptr->GetOutput();
#ifdef DEBUG
      assert(input != nullptr);
      assert(output != nullptr);
#endif
      const size_t buffer_size = ptr->GetBufferSize();
      std::shared_ptr<evaluation::SingleInputWithBufferKernelEval> eval =
          std::make_shared<evaluation::SingleInputWithBufferKernelEval>(
              std::move(generator), buffer_size);
      std::string name = ptr->GetName();
      evaluator_->RegisterEval(std::move(name), std::move(eval));
    } else if (std::shared_ptr<kernel::DoubleInputsWithoutBufferKernelGenerator>
                   generator = std::dynamic_pointer_cast<
                       kernel::DoubleInputsWithoutBufferKernelGenerator>(
                       kgenerator)) {
      std::shared_ptr<flow::DoubleInputsWithoutBufferNode> ptr =
          std::dynamic_pointer_cast<flow::DoubleInputsWithoutBufferNode>(node);
#ifdef DEBUG
      assert(ptr != nullptr);
#endif
      std::shared_ptr<flow::Region> lhs = ptr->GetLhs();
      std::shared_ptr<flow::Region> rhs = ptr->GetRhs();
      std::shared_ptr<flow::Region> output = ptr->GetOutput();
#ifdef DEBUG
      assert(lhs != nullptr);
      assert(rhs != nullptr);
      assert(output != nullptr);
#endif
      std::shared_ptr<evaluation::DoubleInputsWithoutBufferKernelEval> eval =
          std::make_shared<evaluation::DoubleInputsWithoutBufferKernelEval>(
              std::move(generator));
      std::string name = ptr->GetName();
      evaluator_->RegisterEval(std::move(name), std::move(eval));
    } else if (std::shared_ptr<kernel::DoubleInputsWithBufferKernel> kernel =
                   std::dynamic_pointer_cast<
                       kernel::DoubleInputsWithBufferKernel>(kgenerator)) {
#ifdef DEBUG
      assert(false && "unimplemented");
#else
      __builtin_unreachable();
#endif
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
  }
  return evaluator_;
}

} // namespace evaluation
} // namespace cpu_transformers
