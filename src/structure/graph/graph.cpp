#include "structure/graph/graph.h"
#include "fmt/core.h"
#include "structure/graph/edge.h"
#include "utils/isa.hpp"
#include <cassert>
#include <exception>
#include <memory>
#include <optional>
#include <vector>

namespace {
using namespace cpu_transformers::graph;
class NodeEdgeNotMatchException : public std::exception {
public:
  NodeEdgeNotMatchException(const std::string &from, const std::string &to);
  const char *what() const noexcept override;

private:
  const std::string message_;
};

class UnknownEdgeException : public std::exception {
public:
  UnknownEdgeException(const std::string &edge_name);
  UnknownEdgeException(const Edge &edge);
  const char *what() const noexcept override;

private:
  const std::string message_;
};

class UnknownNodeException : public std::exception {
public:
  UnknownNodeException(const std::string &node_name);
  UnknownNodeException(const Node &node);
  const char *what() const noexcept override;

private:
  const std::string message_;
};

NodeEdgeNotMatchException::NodeEdgeNotMatchException(const std::string &from,
                                                     const std::string &to)
    : message_(fmt::format("The relationship of {} and {} is not matched.",
                           from, to)) {}

const char *NodeEdgeNotMatchException::what() const noexcept {
  return message_.c_str();
}

UnknownEdgeException::UnknownEdgeException(const std::string &edge_name)
    : message_(fmt::format("Try to access an unknown edge {}.", edge_name)) {}
UnknownEdgeException::UnknownEdgeException(const Edge &edge)
    : UnknownEdgeException(edge.GetName()) {}

const char *UnknownEdgeException::what() const noexcept {
  return message_.c_str();
}

UnknownNodeException::UnknownNodeException(const std::string &node_name)
    : message_(fmt::format("Try to access an unknown node {}.", node_name)) {}
UnknownNodeException::UnknownNodeException(const Node &node) {}

const char *UnknownNodeException::what() const noexcept {
  return message_.c_str();
}
} // namespace

namespace cpu_transformers {
namespace graph {
bool Graph::ExistEdge(const std::string &name) const {
  return edges_.find(name) != edges_.end();
}

bool Graph::AddEdge(std::shared_ptr<Edge> &&edge) {
  std::string name = edge->GetName();
  if (ExistEdge(name)) {
    return false;
  }
  edges_[name] = EdgeContainer{std::move(edge), {}, {}};
  return true;
}

bool Graph::DeleteEdge(const std::string &name) {
  if (!ExistEdge(name)) {
    return false;
  }
  edges_.erase(name);
  return true;
}

std::shared_ptr<Edge> Graph::GetEdge(const std::string &name) const {
  auto it = edges_.find(name);
  if (it == edges_.end()) {
    return nullptr;
  }
  return it->second.ptr;
}

std::shared_ptr<Node> Graph::GetEdgeFrom(const std::string &edge_name) const {
  auto it = edges_.find(edge_name);
  if (it == edges_.end()) {
    throw UnknownEdgeException(edge_name);
  }
  std::optional<std::string> opt = it->second.from;
  if (!opt.has_value()) {
    return nullptr;
  }
  std::string node_name = *opt;
  std::shared_ptr<Node> node = GetNode(node_name);
  return node;
}

std::shared_ptr<Node> Graph::GetEdgeFrom(const Edge &edge) const {
  return GetEdgeFrom(edge.GetName());
}

std::vector<std::shared_ptr<Node>>
Graph::GetEdgeTo(const std::string &edge_name) const {
  auto it = edges_.find(edge_name);
  if (it == edges_.end()) {
    throw UnknownEdgeException(edge_name);
  }
  std::vector<std::shared_ptr<Node>> result;
  for (const std::string &name : it->second.to) {
    result.push_back(GetNode(name));
  }
  return result;
}

std::vector<std::shared_ptr<Node>> Graph::GetEdgeTo(const Edge &edge) const {
  return GetEdgeTo(edge.GetName());
}

bool Graph::ExistNode(const std::string &name) const {
  return nodes_.find(name) != nodes_.end();
}

bool Graph::PutNode(std::shared_ptr<Node> &&node) {
  std::string name = node->GetName();
  if (ExistEdge(name)) {
    return false;
  }
  nodes_[name] = NodeContainer{std::move(node), {}, {}};
  return true;
}

bool Graph::DeleteNode(const std::string &name) {
  if (!ExistNode(name)) {
    return false;
  }
  nodes_.erase(name);
  return true;
}

std::shared_ptr<Node> Graph::GetNode(const std::string &name) const {
  auto it = nodes_.find(name);
  if (it == nodes_.end()) {
    return nullptr;
  }
  return it->second.ptr;
}

std::vector<std::shared_ptr<Edge>>
Graph::GetNodeFrom(const std::string &node_name) const {
  auto it = nodes_.find(node_name);
  if (it == nodes_.end()) {
    throw UnknownNodeException(node_name);
  }
  std::vector<std::shared_ptr<Edge>> result;
  for (const std::string &name : it->second.from) {
    result.push_back(GetEdge(name));
  }
#ifdef DEBUG
  for (const std::shared_ptr<Edge> &edge : result) {
    std::vector<std::shared_ptr<Node>> nodes = GetEdgeTo(*edge);
    bool found = false;
    for (const std::shared_ptr<Node> &node : nodes) {
      if (node->GetName() == node_name) {
        found = true;
      }
    }
    if (!found) {
      throw NodeEdgeNotMatchException(node_name, edge->GetName());
    }
  }
#endif
  return result;
}

std::vector<std::shared_ptr<Edge>> Graph::GetNodeFrom(const Node &node) const {
  return GetNodeFrom(node.GetName());
}

std::vector<std::shared_ptr<Edge>>
Graph::GetNodeTo(const std::string &node_name) const {
  auto it = nodes_.find(node_name);
  if (it == nodes_.end()) {
    throw UnknownNodeException(node_name);
  }
  std::vector<std::shared_ptr<Edge>> result;
  for (const std::string &name : it->second.to) {
    result.push_back(GetEdge(name));
  }
#ifdef DEBUG
  for (const std::shared_ptr<Edge> &edge : result) {
    std::shared_ptr<Node> node = GetEdgeFrom(*edge);
    if (node->GetName() != node_name) {
      throw NodeEdgeNotMatchException(node_name, edge->GetName());
    }
  }
#endif
  return result;
}

std::vector<std::shared_ptr<Edge>> Graph::GetNodeTo(const Node &node) const {
  return GetNodeTo(node.GetName());
}

std::vector<std::shared_ptr<Node>>
Graph::GetNextNodes(const std::string &name) const {
  auto it = nodes_.find(name);
  if (it == nodes_.end()) {
    throw UnknownNodeException(name);
  }
  std::vector<std::shared_ptr<Node>> result;
  for (const std::string &edge_name : it->second.to) {
    std::vector<std::shared_ptr<Node>> nodes = GetEdgeTo(edge_name);
    for (const std::shared_ptr<Node> &node : nodes) {
      result.push_back(node);
    }
  }
  return result;
}

std::vector<std::shared_ptr<Node>> Graph::GetNextNodes(const Node &node) const {
  return GetNextNodes(node.GetName());
}

std::vector<std::string>
Graph::GetNextNodeNames(const std::string &name) const {
  std::vector<std::string> result;
  for (const std::shared_ptr<Node> &node : GetNextNodes(name)) {
    result.push_back(node->GetName());
  }
  return result;
}

std::vector<std::string> Graph::GetNextNodeNames(const Node &node) const {
  return GetNextNodeNames(node.GetName());
}

std::vector<std::shared_ptr<Edge>> Graph::GetAllEdges() const {
  std::vector<std::shared_ptr<Edge>> result;
  for (const auto &pair : edges_) {
    result.push_back(pair.second.ptr);
  }
  return result;
}

std::vector<std::shared_ptr<InputEdge>> Graph::GetInputEdges() const {
  std::vector<std::shared_ptr<InputEdge>> result;
  for (const auto &pair : edges_) {
    if (std::shared_ptr<InputEdge> inputEdge =
            std::dynamic_pointer_cast<InputEdge>(pair.second.ptr)) {
#ifdef DEBUG
      assert(pair.second.from.has_value() == false);
#endif
      result.push_back(std::move(inputEdge));
    }
  }
  return result;
}

std::vector<std::string> Graph::GetInputEdgeNames() const {
  std::vector<std::string> result;
  for (const auto &pair : edges_) {
    if (isa<InputEdge>(pair.second.ptr.get())) {
#ifdef DEBUG
      assert(!pair.second.from.has_value());
#endif
      result.push_back(pair.first);
    }
  }
  return result;
}

std::vector<std::shared_ptr<Node>> Graph::GetAllNodes() const {
  std::vector<std::shared_ptr<Node>> result;
  for (const auto &pair : nodes_) {
    result.push_back(pair.second.ptr);
  }
  return result;
}

std::vector<std::string> Graph::GetAllNodeNames() const {
  std::vector<std::string> result;
  for (const auto &pair : nodes_) {
    result.push_back(pair.first);
  }
  return result;
}

bool Graph::EdgeToNode(const std::string &edge_name,
                       const std::string &node_name) {
  if (!ExistEdge(edge_name) || !ExistNode(node_name)) {
    return false;
  }
  edges_[edge_name].to.push_back(node_name);
  nodes_[node_name].from.push_back(edge_name);
  return true;
}

bool Graph::EdgeToNode(const Edge &edge, const Node &node) {
  return EdgeToNode(edge.GetName(), node.GetName());
}

bool Graph::NodeToEdge(const std::string &node_name,
                       const std::string &edge_name) {
  if (!ExistNode(node_name) || !ExistEdge(edge_name)) {
    return false;
  }
  nodes_[node_name].to.push_back(edge_name);
  edges_[edge_name].from = node_name;
  return true;
}

bool Graph::NodeToEdge(const Node &node, const Edge &edge) {
  return NodeToEdge(node.GetName(), edge.GetName());
}
} // namespace graph
} // namespace cpu_transformers
