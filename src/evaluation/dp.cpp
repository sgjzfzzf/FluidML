#include "evaluation/dp.h"
#include "evaluation/eval.h"
#include "structure/flow/edge.h"
#include "structure/flow/flow.h"
#include "structure/flow/node.h"
#include "structure/flow/region.h"
#include "utils/isa.hpp"
#include "utils/utils.h"
#include "worker/evaluator.h"
#include "worker/utils.h"
#include <limits>
#include <list>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#ifdef DEBUG
#include "fmt/ranges.h"
#include <cassert>
#endif

namespace cpu_transformers {
namespace evaluation {

class SubFlowsBuilder {
public:
  SubFlowsBuilder(const flow::Flow &flow);
  SubFlowsBuilder(const SubFlowsBuilder &dp) = delete;
  SubFlowsBuilder(SubFlowsBuilder &&dp) = default;
  std::vector<flow::Flow> Run();

private:
  struct Information {
    size_t distance;
    std::shared_ptr<flow::Edge> prev;
  };
  size_t findLongestPathTo(std::shared_ptr<flow::Edge> edge);
  const flow::Flow &flow_;
  std::unordered_map<std::string, Information> djikstra_table_;
};

class DPOnNoOverlapFlowWoker {
public:
  DPOnNoOverlapFlowWoker(const flow::Flow &flow,
                         std::shared_ptr<worker::Evaluator> &&evaluator);
  DPOnNoOverlapFlowWoker(const DPOnNoOverlapFlowWoker &runner) = delete;
  DPOnNoOverlapFlowWoker(DPOnNoOverlapFlowWoker &&runner) = default;
  DynamicProgrammingPlan Run();

private:
  struct EdgeLayout {
    std::shared_ptr<flow::Edge> edge;
    std::vector<size_t> layout;
  };
  struct EdgeLayoutEqual {
    bool operator()(const EdgeLayout &lhs, const EdgeLayout &rhs) const;
  };
  struct EdgeLayoutHash {
    static constexpr size_t kHashSeed = 0x9e3779b9;
    size_t operator()(const EdgeLayout &edge) const;
  };
  size_t runOn(std::shared_ptr<flow::Edge> edge,
               const std::vector<size_t> &layout);
  const flow::Flow &flow_;
  std::shared_ptr<worker::Evaluator> evaluator_;
  std::unordered_map<EdgeLayout, std::tuple<size_t, std::vector<EdgeLayout>>,
                     EdgeLayoutHash, EdgeLayoutEqual>
      dp_table_;
};

class DynamicProgrammingTableImpl : public DynamicProgrammingTable {
public:
  DynamicProgrammingTableImpl(const flow::Flow &flow);
  DynamicProgrammingTableImpl(const DynamicProgrammingTableImpl &table) =
      delete;
  DynamicProgrammingTableImpl(DynamicProgrammingTableImpl &&table) = default;
  DynamicProgrammingPlan Run() override;

private:
  std::shared_ptr<worker::Evaluator> getEvaluator();
  const flow::Flow &flow_;
  std::shared_ptr<worker::Evaluator> evaluator_;
};

std::shared_ptr<DynamicProgrammingTable>
DynamicProgrammingTable::Make(const flow::Flow &flow) {
  return std::make_shared<DynamicProgrammingTableImpl>(flow);
}

SubFlowsBuilder::SubFlowsBuilder(const flow::Flow &flow) : flow_(flow) {}

std::vector<flow::Flow> SubFlowsBuilder::Run() {
  std::vector<std::shared_ptr<flow::Edge>> edges = flow_.GetEdges();
  std::vector<std::shared_ptr<flow::Node>> nodes = flow_.GetNodes();
  size_t farthest_distance = 0;
  std::shared_ptr<flow::Edge> farthest_edge = nullptr;
  std::vector<flow::Flow> subflows;
  for (const std::shared_ptr<flow::Edge> &edge : edges) {
#ifdef DEBUG
    assert(edge != nullptr);
#endif
    size_t distance = findLongestPathTo(edge);
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
       edge = djikstra_table_[edge->GetName()].prev) {
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
                                                     flow_.GetEdges(lhs_name),
                                                 rhs_edges =
                                                     flow_.GetEdges(rhs_name);
        std::shared_ptr<flow::Edge> lhs_edge =
                                        flow_.GetLhsEdge(*double_inputs_node),
                                    rhs_edge =
                                        flow_.GetRhsEdge(*double_inputs_node),
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
                                        flow_.GetInputEdge(*single_input_node),
                                    output_edge =
                                        flow_.GetOutputEdge(*single_input_node);
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
                                        flow_.GetLhsEdge(*double_inputs_node),
                                    rhs_edge =
                                        flow_.GetRhsEdge(*double_inputs_node),
                                    output_edge = flow_.GetOutputEdge(
                                        *double_inputs_node);
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
            flow_.GetInputEdge(*single_input_node);
        std::shared_ptr<flow::OwnFromEdge> output_edge =
            flow_.GetOutputEdge(*single_input_node);
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
        } else if (std::shared_ptr<flow::InputEdge> input_input_edge =
                       std::dynamic_pointer_cast<flow::InputEdge>(input_edge)) {
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
        std::shared_ptr<flow::OwnToEdge> lhs_edge = flow_.GetLhsEdge(
                                             *double_inputs_node),
                                         rhs_edge = flow_.GetRhsEdge(
                                             *double_inputs_node);
        std::shared_ptr<flow::OwnFromEdge> output_edge =
            flow_.GetOutputEdge(*double_inputs_node);
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
        std::shared_ptr<flow::DoubleInputsNode> real_node = nullptr;
        if (std::shared_ptr<flow::MemoryEdge> lhs_memory_edge =
                std::dynamic_pointer_cast<flow::MemoryEdge>(lhs_edge)) {
          std::shared_ptr<flow::Node> from = lhs_memory_edge->GetFrom();
          if (branch_nodes.find(from) != branch_nodes.end()) {
            real_lhs_edge = lhs_edge;
          } else {
            std::shared_ptr<flow::DoubleInputsNode> node_clone =
                double_inputs_node;
            real_lhs_edge = std::make_shared<flow::OwnToEdge>(
                lhs_edge->GetRegion(), std::move(node_clone));
          }
        } else if (std::shared_ptr<flow::InputEdge> lhs_input_edge =
                       std::dynamic_pointer_cast<flow::InputEdge>(lhs_edge)) {
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
            real_rhs_edge = std::make_shared<flow::OwnToEdge>(
                rhs_edge->GetRegion(), std::move(node_clone));
          }
        } else if (std::shared_ptr<flow::InputEdge> rhs_input_edge =
                       std::dynamic_pointer_cast<flow::InputEdge>(rhs_edge)) {
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
            real_output_edge = std::make_shared<flow::OwnFromEdge>(
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
        branch_subflow.PutNode(std::move(real_node));
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
    SubFlowsBuilder builder(branch_subflow);
    std::vector<flow::Flow> flows = builder.Run();
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

size_t SubFlowsBuilder::findLongestPathTo(std::shared_ptr<flow::Edge> edge) {
  auto it = djikstra_table_.find(edge->GetName());
  if (it != djikstra_table_.end()) {
    auto [distance, _] = it->second;
    return distance;
  }
  size_t distance = -1;
  std::shared_ptr<flow::Edge> prev = nullptr;
  if (std::shared_ptr<flow::InputEdge> input_edge =
          std::dynamic_pointer_cast<flow::InputEdge>(edge)) {
    distance = 0;
  } else if (std::shared_ptr<flow::OwnFromEdge> own_from_edge =
                 std::dynamic_pointer_cast<flow::OwnFromEdge>(edge)) {
    std::shared_ptr<flow::Node> from = own_from_edge->GetFrom();
    if (std::shared_ptr<flow::SingleInputNode> single_input_node =
            std::dynamic_pointer_cast<flow::SingleInputNode>(from)) {
      const std::string &input_name = single_input_node->GetInputAsString();
      std::vector<std::shared_ptr<flow::Edge>> edges =
          flow_.GetEdges(input_name);
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
      distance = findLongestPathTo(input) + 1;
      prev = std::move(input);
    } else if (std::shared_ptr<flow::DoubleInputsNode> double_inputs_node =
                   std::dynamic_pointer_cast<flow::DoubleInputsNode>(from)) {
      const std::string &lhs_name = double_inputs_node->GetLhsAsString(),
                        &rhs_name = double_inputs_node->GetRhsAsString();
      const std::vector<std::shared_ptr<flow::Edge>> &lhs_edges =
                                                         flow_.GetEdges(
                                                             lhs_name),
                                                     &rhs_edges =
                                                         flow_.GetEdges(
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
      const size_t lhs_distance = findLongestPathTo(lhs),
                   rhs_distance = findLongestPathTo(rhs);
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
  djikstra_table_.insert({edge->GetName(), {distance, prev}});
  return distance;
}

DPOnNoOverlapFlowWoker::DPOnNoOverlapFlowWoker(
    const flow::Flow &flow, std::shared_ptr<worker::Evaluator> &&evaluator)
    : flow_(flow), evaluator_(std::move(evaluator)) {}

DynamicProgrammingPlan DPOnNoOverlapFlowWoker::Run() {
  std::vector<std::shared_ptr<flow::Edge>> edges = flow_.GetEdges();
  std::unordered_map<std::string, std::vector<size_t>> layout_table;
  for (std::shared_ptr<flow::Edge> edge : edges) {
    if (isa<flow::OutputEdge>(edge)) {
      const std::string &name = edge->GetName();
      const Meta &meta = edge->GetMeta();
      const std::vector<size_t> &layout = edge->GetLayout();
      runOn(edge, layout);
      std::list<std::tuple<std::shared_ptr<flow::Edge>, std::vector<size_t>>>
          queue = {{edge, layout}};
      layout_table.insert({name, layout});
      while (!queue.empty()) {
        auto [edge, layout] = std::move(queue.front());
        queue.pop_front();
        auto it = dp_table_.find({edge, layout});
#ifdef DEBUG
        assert(it != dp_table_.end());
#endif
        auto [_, deps] = it->second;
        for (const EdgeLayout &dep : deps) {
#ifdef DEBUG
          assert(dep.edge != nullptr);
#endif
          const std::string &name = dep.edge->GetName();
#ifdef DEBUG
          auto cannot_exist_it = dp_table_.find(dep);
          assert(cannot_exist_it != dp_table_.end());
#endif
          layout_table.insert({name, dep.layout});
          queue.push_back({dep.edge, dep.layout});
        }
      }
    }
  }
  return DynamicProgrammingPlan(std::move(layout_table));
}

bool DPOnNoOverlapFlowWoker::EdgeLayoutEqual::operator()(
    const EdgeLayout &lhs, const EdgeLayout &rhs) const {
  return lhs.edge == rhs.edge && lhs.layout == rhs.layout;
}

size_t DPOnNoOverlapFlowWoker::EdgeLayoutHash::operator()(
    const EdgeLayout &edge) const {
  size_t hash = 0;
  std::hash<std::shared_ptr<flow::Edge>> edge_hash;
  std::hash<int64_t> layout_hash;
  hash ^= edge_hash(edge.edge);
  for (int64_t i : edge.layout) {
    hash ^= layout_hash(i) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  return hash;
}

size_t DPOnNoOverlapFlowWoker::runOn(std::shared_ptr<flow::Edge> edge,
                                     const std::vector<size_t> &layout) {
  auto it = dp_table_.find({edge, layout});
  if (it != dp_table_.end()) {
    return std::get<0>(it->second);
  }
  if (std::shared_ptr<flow::InputEdge> input_edge =
          std::dynamic_pointer_cast<flow::InputEdge>(edge)) {
    constexpr size_t kInputEdgeTimeCost = 0;
    std::tuple<size_t, std::vector<EdgeLayout>> result = {kInputEdgeTimeCost,
                                                          {}};
    dp_table_.insert({{edge, layout}, result});
    return kInputEdgeTimeCost;
  } else if (std::shared_ptr<flow::OwnFromEdge> own_from_edge =
                 std::dynamic_pointer_cast<flow::OwnFromEdge>(edge)) {
    std::shared_ptr<flow::Node> node = own_from_edge->GetFrom();
    const std::string &node_name = node->GetName();
    if (std::shared_ptr<flow::SingleInputNode> single_input_node =
            std::dynamic_pointer_cast<flow::SingleInputNode>(node)) {
      std::shared_ptr<flow::OwnToEdge> input_edge =
          flow_.GetInputEdge(*single_input_node);
      const Meta &meta = input_edge->GetMeta();
      const std::vector<int64_t> &shape = meta.GetShape();
      const size_t shape_len = shape.size();
      std::vector<std::vector<size_t>> input_layouts =
          utils::GenAllOrders(shape_len);
      size_t min_time_cost = std::numeric_limits<size_t>::max();
      std::vector<EdgeLayout> deps;
      for (const std::vector<size_t> &prev_layout : input_layouts) {
        evaluation::SingleInputKernelEval &eval =
            evaluator_->GetSingleInputEval(node_name);
        const size_t time_cost = eval.GetTimeCost(prev_layout, layout) +
                                 runOn(input_edge, prev_layout);
        if (min_time_cost > time_cost) {
          min_time_cost = time_cost;
          deps = {{input_edge, prev_layout}};
        }
      }
      dp_table_.insert({{edge, layout}, {min_time_cost, std::move(deps)}});
      return min_time_cost;
    } else if (std::shared_ptr<flow::DoubleInputsNode> double_inputs_node =
                   std::dynamic_pointer_cast<flow::DoubleInputsNode>(node)) {
      std::shared_ptr<flow::OwnToEdge> lhs_edge = flow_.GetLhsEdge(
                                           *double_inputs_node),
                                       rhs_edge = flow_.GetRhsEdge(
                                           *double_inputs_node);
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
      std::vector<EdgeLayout> deps;
      for (const std::vector<size_t> &lhs_layout : lhs_layouts) {
        for (const std::vector<size_t> &rhs_layout : rhs_layouts) {
          evaluation::DoubleInputsKernelEval &eval =
              evaluator_->GetDoubleInputsEval(node_name);
          const size_t time_cost =
              eval.GetTimeCost(lhs_layout, rhs_layout, layout) +
              runOn(lhs_edge, lhs_layout) + runOn(rhs_edge, rhs_layout);
          if (min_time_cost > time_cost) {
            min_time_cost = time_cost;
            deps = {{lhs_edge, lhs_layout}, {rhs_edge, rhs_layout}};
          }
        }
      }
      dp_table_.insert({{edge, layout}, {min_time_cost, std::move(deps)}});
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

#ifdef DEBUG
std::ostream &operator<<(std::ostream &os, const DynamicProgrammingPlan &plan) {
  os << fmt::format("{}", plan.plan_);
  return os;
}
#endif

DynamicProgrammingTableImpl::DynamicProgrammingTableImpl(const flow::Flow &flow)
    : flow_(flow) {}

DynamicProgrammingPlan DynamicProgrammingTableImpl::Run() {
  SubFlowsBuilder builder(flow_);
  std::vector<flow::Flow> subflows = builder.Run();
  DynamicProgrammingPlan plan;
  for (const flow::Flow &subflow : subflows) {
#ifdef DEBUG
    assert(subflow.IsNoOverlapFlow());
#endif
    std::shared_ptr<worker::Evaluator> evaluator = getEvaluator();
    DPOnNoOverlapFlowWoker worker(subflow, std::move(evaluator));
    plan = Merge(plan, worker.Run());
  }
  return plan;
}

std::shared_ptr<worker::Evaluator> DynamicProgrammingTableImpl::getEvaluator() {
  if (evaluator_ != nullptr) {
    return evaluator_;
  }
  evaluator_ = worker::Evaluator::Make();
  const std::vector<std::shared_ptr<flow::Node>> &nodes = flow_.GetNodes();
  for (std::shared_ptr<flow::Node> node : nodes) {
    std::shared_ptr<kernel::Kernel> k = worker::SelectKernel(node.get());
    size_t time_cost = 0;
    if (std::shared_ptr<kernel::SingleInputWithoutBufferKernel> kernel =
            std::dynamic_pointer_cast<kernel::SingleInputWithoutBufferKernel>(
                k)) {
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
      Meta input_meta = input->GetMeta();
      Meta output_meta = output->GetMeta();
      std::shared_ptr<evaluation::SingleInputWithoutBufferKernelEval> eval =
          std::make_shared<evaluation::SingleInputWithoutBufferKernelEval>(
              std::move(kernel), std::move(input_meta), std::move(output_meta));
      std::string name = ptr->GetName();
      evaluator_->RegisterEval(std::move(name), std::move(eval));
    } else if (std::shared_ptr<kernel::SingleInputWithBufferKernel> kernel =
                   std::dynamic_pointer_cast<
                       kernel::SingleInputWithBufferKernel>(k)) {
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
      Meta input_meta = input->GetMeta();
      Meta output_meta = output->GetMeta();
      const size_t buffer_size = ptr->GetBufferSize();
      std::shared_ptr<evaluation::SingleInputWithBufferKernelEval> eval =
          std::make_shared<evaluation::SingleInputWithBufferKernelEval>(
              std::move(kernel), std::move(input_meta), std::move(output_meta),
              buffer_size);
      std::string name = ptr->GetName();
      evaluator_->RegisterEval(std::move(name), std::move(eval));
    } else if (std::shared_ptr<kernel::DoubleInputsWithoutBufferKernel> kernel =
                   std::dynamic_pointer_cast<
                       kernel::DoubleInputsWithoutBufferKernel>(k)) {
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
      Meta lhs_meta = lhs->GetMeta();
      Meta rhs_meta = rhs->GetMeta();
      Meta output_meta = output->GetMeta();
      std::shared_ptr<evaluation::DoubleInputsWithoutBufferKernelEval> eval =
          std::make_shared<evaluation::DoubleInputsWithoutBufferKernelEval>(
              std::move(kernel), std::move(lhs_meta), std::move(rhs_meta),
              std::move(output_meta));
      std::string name = ptr->GetName();
      evaluator_->RegisterEval(std::move(name), std::move(eval));
    } else if (std::shared_ptr<kernel::DoubleInputsWithBufferKernel> kernel =
                   std::dynamic_pointer_cast<
                       kernel::DoubleInputsWithBufferKernel>(k)) {
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
