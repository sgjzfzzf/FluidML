#include "structure/graph/edge.h"
#include "structure/graph/graph.h"
#include "structure/tensor/meta.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace graph {

Edge::Edge(std::string &&name, Graph *graph) : name_(name), graph_(graph) {}

const std::string &Edge::GetName() const { return name_; }

Graph *Edge::GetGraph() const { return graph_; }

std::shared_ptr<Node> Edge::GetInputNode() const {
  return graph_->GetEdgeFrom(*this);
}

std::vector<std::shared_ptr<Node>> Edge::GetOutputNodes() const {
  return graph_->GetEdgeTo(*this);
}

void Edge::Delete() {
  if (graph_) {
#ifdef DEBUG
    assert(
#endif
        graph_->DeleteEdge(*this)
#ifdef DEBUG
    )
#endif
        ;
  }
}

void Edge::ClearInput(Node &node) {
  if (graph_) {
    graph_->ClearEdgeToNode(*this, node);
  }
}

void Edge::ClearInput(const std::string &name) {
  if (graph_) {
    graph_->ClearEdgeToNode(name, name_);
  }
}

void Edge::ClearOutput(Node &node) {
  if (graph_) {
    graph_->ClearNodeToEdge(node, *this);
  }
}

void Edge::ClearOutput(const std::string &name) {
  if (graph_) {
    graph_->ClearNodeToEdge(name_, name);
  }
}

void Edge::ClearInputs() {
  if (graph_) {
    graph_->ClearEdgeFrom(*this);
  }
}

void Edge::ClearOutputs() {
  if (graph_) {
    graph_->ClearEdgeTos(*this);
  }
}

void Edge::PutInput(Node &node) {
  if (graph_) {
    graph_->NodeToEdge(node, *this);
  }
}

void Edge::PutOutput(Node &node) {
  if (graph_) {
    graph_->EdgeToNode(*this, node);
  }
}

ConstantEdge::ConstantEdge(std::string &&name) : Edge(std::move(name)) {}

ConstantScalarEdge::ConstantScalarEdge(std::string &&name, Type type,
                                       float64_t scalar)
    : ConstantEdge(std::move(name)), type_(type), scalar_(scalar) {}

Type ConstantScalarEdge::GetType() const { return type_; }

const std::vector<int64_t> &ConstantScalarEdge::GetShape() const {
  static const std::vector<int64_t> shape = {};
  return shape;
}

float64_t ConstantScalarEdge::GetValue() const { return scalar_; }

ConstantTensorEdge::ConstantTensorEdge(std::string &&name, Type type,
                                       Tensor &&tensor)
    : ConstantEdge(std::move(name)), tensor_(std::move(tensor)) {}

const Meta &ConstantTensorEdge::GetMeta() const { return tensor_.GetMeta(); }

Type ConstantTensorEdge::GetType() const { return tensor_.GetType(); }

const std::vector<int64_t> &ConstantTensorEdge::GetShape() const {
  return tensor_.GetShape();
}

const Tensor &ConstantTensorEdge::GetValue() const { return tensor_; }

NonConstantEdge::NonConstantEdge(std::string &&name, Type type,
                                 std::vector<int64_t> &&shape)
    : Edge(std::move(name)), meta_(type, std::move(shape)) {}

const Meta &NonConstantEdge::GetMeta() const { return meta_; }

Type NonConstantEdge::GetType() const { return GetMeta().GetType(); }

const std::vector<int64_t> &NonConstantEdge::GetShape() const {
  return GetMeta().GetShape();
}

PureEdge::PureEdge(std::string &&name, Type type, std::vector<int64_t> &&shape)
    : NonConstantEdge(std::move(name), type, std::move(shape)) {}

InputEdge::InputEdge(std::string &&name, Type type,
                     std::vector<int64_t> &&shape)
    : NonConstantEdge(std::move(name), type, std::move(shape)) {}

OutputEdge::OutputEdge(std::string &&name, Type type,
                       std::vector<int64_t> &&shape)
    : NonConstantEdge(std::move(name), type, std::move(shape)) {}

} // namespace graph
} // namespace cpu_transformers
