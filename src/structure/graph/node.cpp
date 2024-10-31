#include "structure/graph/node.h"
#include "structure/graph/attribute.h"
#include "structure/graph/graph.h"
#include <unordered_map>

namespace fluidml {
namespace graph {

Node::Node(std::string &&name, Op op,
           std::unordered_map<std::string, Attribute> &&attributes,
           Graph *graph)
    : name_(std::move(name)), op_(op), attributes_(attributes), graph_(graph) {}

const std::string &Node::GetName() const { return name_; }

const Node::Op &Node::GetOp() const { return op_; }

bool Node::HasAttribute(const std::string &name) const {
  return attributes_.find(name) != attributes_.end();
}

const Attribute &Node::GetAttribute(const std::string &name) const {
  return attributes_.at(name);
}

const std::unordered_map<std::string, Attribute> &Node::GetAttributes() const {
  return attributes_;
}

Graph *Node::GetGraph() const { return graph_; }

std::vector<std::shared_ptr<Edge>> Node::GetInputEdges() const {
  return graph_ ? graph_->GetNodeFrom(name_)
                : std::vector<std::shared_ptr<Edge>>{};
}

std::vector<std::shared_ptr<Edge>> Node::GetOutputEdges() const {
  return graph_ ? graph_->GetNodeTo(name_)
                : std::vector<std::shared_ptr<Edge>>{};
}

std::vector<std::shared_ptr<Node>> Node::GetInputNodes() const {
  std::vector<std::shared_ptr<Node>> nodes;
  if (graph_) {
    std::vector<std::shared_ptr<Edge>> input_edges = GetInputEdges();
    for (std::shared_ptr<Edge> edge : input_edges) {
      nodes.push_back(graph_->GetEdgeFrom(*edge));
    }
  }
  return nodes;
}

std::vector<std::shared_ptr<Node>> Node::GetOutputNodes() const {
  std::vector<std::shared_ptr<Node>> nodes;
  if (graph_) {
    std::vector<std::shared_ptr<Edge>> output_edges = GetOutputEdges();
    for (std::shared_ptr<Edge> edge : output_edges) {
      std::vector<std::shared_ptr<Node>> output_nodes =
          graph_->GetEdgeTo(*edge);
      nodes.insert(nodes.end(), output_nodes.begin(), output_nodes.end());
    }
  }
  return nodes;
}

void Node::Delete() {
  if (graph_) {
#ifdef DEBUG
    assert(
#endif
        graph_->DeleteNode(name_)
#ifdef DEBUG
    )
#endif
        ;
  }
}

void Node::ClearInput(const Edge &edge) {
  if (graph_) {
    graph_->ClearEdgeToNode(edge, *this);
  }
}

void Node::ClearInput(const std::string &name) {
  if (graph_) {
    graph_->ClearEdgeToNode(name, name_);
  }
}

void Node::ClearOutput(const Edge &edge) {
  if (graph_) {
    graph_->ClearNodeToEdge(*this, edge);
  }
}

void Node::ClearOutput(const std::string &name) {
  if (graph_) {
    graph_->ClearNodeToEdge(name_, name);
  }
}

void Node::ClearInputs() {
  if (graph_) {
    graph_->ClearNodeFrom(name_);
  }
}

void Node::ClearOutputs() {
  if (graph_) {
    graph_->ClearNodeTos(name_);
  }
}

void Node::PutInput(Edge &edge) {
  if (graph_) {
    graph_->EdgeToNode(edge, *this);
  }
}

void Node::PutOutput(Edge &edge) {
  if (graph_) {
    graph_->NodeToEdge(*this, edge);
  }
}

} // namespace graph
} // namespace fluidml
