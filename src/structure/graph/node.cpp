#include "structure/graph/node.h"
#include "structure/graph/attribute.h"
#include <unordered_map>

namespace cpu_transformers {
namespace graph {
Node::Node(std::string &&name, Op op) : name_(std::move(name)), op_(op) {}

Node::Node(std::string &&name, Op op,
           std::unordered_map<std::string, Attribute> &&attributes)
    : name_(std::move(name)), op_(op), attributes_(attributes) {}

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
} // namespace graph
} // namespace cpu_transformers
