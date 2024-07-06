#include "structure/flow/flow.h"
#include "structure/flow/edge.h"
#include "structure/flow/node.h"
#include "structure/flow/region.h"
#include "utils/isa.hpp"
#include <memory>
#include <string>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace flow {

void Flow::PutNode(std::shared_ptr<Node> &&node) {
  std::string name = node->GetName();
#ifdef DEBUG
  assert(node != nullptr);
  assert(GetNode(name) == nullptr);
#endif
  nodes_.insert({std::move(name), std::move(node)});
}

void Flow::PutEdge(std::shared_ptr<Edge> &&edge) {
#ifdef DEBUG
  assert(edge != nullptr);
  if (std::shared_ptr<MemoryEdge> memory_edge =
          std::dynamic_pointer_cast<MemoryEdge>(edge)) {
    std::vector<std::shared_ptr<Edge>> edges = GetEdges();
    for (const std::shared_ptr<Edge> &medge : edges) {
      if (std::shared_ptr<MemoryEdge> memory_medge =
              std::dynamic_pointer_cast<MemoryEdge>(medge)) {
        assert(memory_medge->GetFrom() != memory_edge->GetFrom() ||
               memory_medge->GetTo() != memory_edge->GetTo());
      }
    }
  }
#endif
  std::string name = edge->GetName();
  edges_.insert({std::move(name), std::move(edge)});
}

void Flow::PutRegion(std::shared_ptr<Region> &&region) {
  std::string name = region->GetName();
#ifdef DEBUG
  assert(region != nullptr);
  assert(GetRegion(name) == nullptr);
#endif
  regions_.insert({std::move(name), std::move(region)});
}

std::shared_ptr<Node> Flow::GetNode(const std::string &name) const {
  auto it = nodes_.find(name);
  return it != nodes_.end() ? it->second : nullptr;
}

std::shared_ptr<Edge> Flow::GetEdge(const std::string &name) const {
  std::vector<std::shared_ptr<Edge>> edges = GetEdges(name);
  const size_t size = edges.size();
  if (size == 1) {
    return edges[0];
  } else if (size == 0) {
    return nullptr;
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
}

std::shared_ptr<Region> Flow::GetRegion(const std::string &name) const {
  auto it = regions_.find(name);
  return it != regions_.end() ? it->second : nullptr;
}

std::vector<std::shared_ptr<Node>> Flow::GetNodes() const {
  std::vector<std::shared_ptr<Node>> nodes;
  for (auto &[k, v] : nodes_) {
    nodes.push_back(v);
  }
  return nodes;
}

std::vector<std::shared_ptr<Edge>>
Flow::GetEdges(const std::string &name) const {
  std::vector<std::shared_ptr<Edge>> edges;
  auto range = edges_.equal_range(name);
  for (auto it = range.first; it != range.second; ++it) {
    edges.push_back(it->second);
  }
  return edges;
}

std::vector<std::shared_ptr<Edge>> Flow::GetEdges() const {
  std::vector<std::shared_ptr<Edge>> edges;
  for (auto &[k, v] : edges_) {
    edges.push_back(v);
  }
  return edges;
}

std::vector<std::shared_ptr<Region>> Flow::GetRegions() const {
  std::vector<std::shared_ptr<Region>> regions;
  for (auto &[k, v] : regions_) {
    regions.push_back(v);
  }
  return regions;
}

std::shared_ptr<OwnToEdge>
Flow::GetInputEdge(const SingleInputNode &node) const {
  const std::string &input_name = node.GetInputAsString();
  std::vector<std::shared_ptr<Edge>> edges = GetEdges(input_name);
  for (std::shared_ptr<Edge> edge : edges) {
    if (std::shared_ptr<OwnToEdge> own_to_edge =
            std::dynamic_pointer_cast<OwnToEdge>(edge)) {
      if (own_to_edge->GetTo().get() == &node) {
        return own_to_edge;
      }
    }
  }
  return nullptr;
}

std::shared_ptr<OwnFromEdge>
Flow::GetOutputEdge(const SingleInputNode &node) const {
  const std::string &output_name = node.GetOutputAsString();
  std::vector<std::shared_ptr<Edge>> edges = GetEdges(output_name);
  for (std::shared_ptr<Edge> edge : edges) {
    if (std::shared_ptr<OwnFromEdge> own_from_edge =
            std::dynamic_pointer_cast<OwnFromEdge>(edge)) {
      if (own_from_edge->GetFrom().get() == &node) {
        return own_from_edge;
      }
    }
  }
  return nullptr;
}

std::shared_ptr<OwnToEdge>
Flow::GetLhsEdge(const DoubleInputsNode &node) const {
  const std::string &lhs_name = node.GetLhsAsString();
  std::vector<std::shared_ptr<Edge>> edges = GetEdges(lhs_name);
  for (std::shared_ptr<Edge> edge : edges) {
    if (std::shared_ptr<OwnToEdge> own_to_edge =
            std::dynamic_pointer_cast<OwnToEdge>(edge)) {
      if (own_to_edge->GetTo().get() == &node) {
        return own_to_edge;
      }
    }
  }
  return nullptr;
}

std::shared_ptr<OwnToEdge>
Flow::GetRhsEdge(const DoubleInputsNode &node) const {
  const std::string &rhs_name = node.GetRhsAsString();
  std::vector<std::shared_ptr<Edge>> edges = GetEdges(rhs_name);
  for (std::shared_ptr<Edge> edge : edges) {
    if (std::shared_ptr<OwnToEdge> own_to_edge =
            std::dynamic_pointer_cast<OwnToEdge>(edge)) {
      if (own_to_edge->GetTo().get() == &node) {
        return own_to_edge;
      }
    }
  }
  return nullptr;
}

std::shared_ptr<OwnFromEdge>
Flow::GetOutputEdge(const DoubleInputsNode &node) const {
  const std::string &output_name = node.GetOutputAsString();
  std::vector<std::shared_ptr<Edge>> edges = GetEdges(output_name);
  for (std::shared_ptr<Edge> edge : edges) {
    if (std::shared_ptr<OwnFromEdge> own_from_edge =
            std::dynamic_pointer_cast<OwnFromEdge>(edge)) {
      if (own_from_edge->GetFrom().get() == &node) {
        return own_from_edge;
      }
    }
  }
  return nullptr;
}

#ifdef DEBUG
// This method is used to check the integrity of the flow. For every node, it
// checks whether the input and output edges all exist in the flow. For every
// edge, it checks whether the from and to nodes all exist in the flow. If any
// of the above conditions is not satisfied, the method returns false.
bool Flow::Check() const {
  std::vector<std::shared_ptr<Node>> nodes = GetNodes();
  std::vector<std::shared_ptr<Edge>> edges = GetEdges();
  for (std::shared_ptr<Node> node : nodes) {
    if (std::shared_ptr<SingleInputNode> single_input_node =
            std::dynamic_pointer_cast<SingleInputNode>(node)) {
      const std::string &input_name = single_input_node->GetInputAsString(),
                        &output_name = single_input_node->GetOutputAsString();
      std::vector<std::shared_ptr<flow::Edge>> input_edges =
                                                   GetEdges(input_name),
                                               output_edges =
                                                   GetEdges(output_name);
      bool found_input = false, found_output = false;
      for (std::shared_ptr<flow::Edge> input_edge : input_edges) {
        if (std::shared_ptr<OwnToEdge> own_to_edge =
                std::dynamic_pointer_cast<OwnToEdge>(input_edge)) {
          if (own_to_edge->GetTo() == node) {
            found_input = true;
            break;
          }
        }
      }
      for (std::shared_ptr<flow::Edge> output_edge : output_edges) {
        if (std::shared_ptr<OwnFromEdge> own_from_edge =
                std::dynamic_pointer_cast<OwnFromEdge>(output_edge)) {
          if (own_from_edge->GetFrom() == node) {
            found_output = true;
            break;
          }
        }
      }
      if (!found_input || !found_output) {
        return false;
      }
    } else if (std::shared_ptr<DoubleInputsNode> double_inputs_node =
                   std::dynamic_pointer_cast<DoubleInputsNode>(node)) {
      const std::string &lhs_name = double_inputs_node->GetLhsAsString(),
                        &rhs_name = double_inputs_node->GetRhsAsString(),
                        &output_name = double_inputs_node->GetOutputAsString();
      std::vector<std::shared_ptr<flow::Edge>> lhs_edges = GetEdges(lhs_name),
                                               rhs_edges = GetEdges(rhs_name),
                                               output_edges =
                                                   GetEdges(output_name);
      bool found_lhs = false, found_rhs = false, found_output = false;
      for (std::shared_ptr<flow::Edge> lhs_edge : lhs_edges) {
        if (std::shared_ptr<OwnToEdge> own_to_edge =
                std::dynamic_pointer_cast<OwnToEdge>(lhs_edge)) {
          if (own_to_edge->GetTo() == node) {
            found_lhs = true;
            break;
          }
        }
      }
      for (std::shared_ptr<flow::Edge> rhs_edge : rhs_edges) {
        if (std::shared_ptr<OwnToEdge> own_to_edge =
                std::dynamic_pointer_cast<OwnToEdge>(rhs_edge)) {
          if (own_to_edge->GetTo() == node) {
            found_rhs = true;
            break;
          }
        }
      }
      for (std::shared_ptr<flow::Edge> output_edge : output_edges) {
        if (std::shared_ptr<OwnFromEdge> own_from_edge =
                std::dynamic_pointer_cast<OwnFromEdge>(output_edge)) {
          if (own_from_edge->GetFrom() == node) {
            found_output = true;
            break;
          }
        }
      }
      if (!found_lhs || !found_rhs || !found_output) {
        return false;
      }
    } else {
      return false;
    }
  }
  for (std::shared_ptr<Edge> edge : edges) {
    std::shared_ptr<Region> region = edge->GetRegion();
    if (region == nullptr) {
      return false;
    }
    if (std::shared_ptr<OwnFromEdge> own_from_edge =
            std::dynamic_pointer_cast<OwnFromEdge>(edge)) {
      std::shared_ptr<Node> from = own_from_edge->GetFrom();
      if (from == nullptr) {
        return false;
      }
      std::string from_name = from->GetName();
      if (GetNode(from_name) == nullptr) {
        return false;
      }
    }
    if (std::shared_ptr<OwnToEdge> own_to_edge =
            std::dynamic_pointer_cast<OwnToEdge>(edge)) {
      std::shared_ptr<Node> to = own_to_edge->GetTo();
      if (to == nullptr) {
        return false;
      }
      std::string to_name = to->GetName();
      if (GetNode(to_name) == nullptr) {
        return false;
      }
    }
  }
  return true;
}

// This method is used to check whether the flow is a no-overlap flow. A
// no-overlap means for every edge in the graph, there is no overlap between the
// dependencies of the edge.
// For example, a graph with edges:
// A -> B, A -> C, B -> D, C -> D
// has overlap, because for edges B and C, they don't have the relationship of
// success, but they shared the same dependency A.
// For another example, a graph with edges:
// A -> B, B -> C, C -> D
// has no overlap, because for every edge, there is no overlap between the
// dependencies.
// Only no-overlap flow can be handled by the DP algorithm.
// The way to check whether a flow is a no-overlap flow is to check whether for
// every memory edge, which edge doesn't interact with the outside world of the
// function, has a unique name in the whole flow. This is because if there is an
// overlap, there must be a node that provides at least two output edges with
// the same region, which means that they share the same name.
bool Flow::IsNoOverlapFlow() const {
  std::vector<std::shared_ptr<Edge>> edges = GetEdges();
  for (std::shared_ptr<Edge> edge : edges) {
    if (std::shared_ptr<MemoryEdge> own_from_edge =
            std::dynamic_pointer_cast<MemoryEdge>(edge)) {
      const std::string &name = own_from_edge->GetName();
      size_t counter = 0;
      std::vector<std::shared_ptr<Edge>> named_edges = GetEdges(name);
      for (std::shared_ptr<Edge> named_edge : named_edges) {
        if (isa<MemoryEdge>(named_edge)) {
          ++counter;
        }
      }
      if (counter != 1) {
        return false;
      }
    }
  }
  return true;
}
#endif

} // namespace flow
} // namespace cpu_transformers
