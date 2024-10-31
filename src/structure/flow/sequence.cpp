#include "structure/flow/sequence.h"
#include "structure/flow/node.h"

namespace fluidml {
namespace flow {

void Sequence::PutNode(std::shared_ptr<Node> &&node) {
  nodes_.push_back(std::move(node));
}

void Sequence::PutEdge(std::shared_ptr<Edge> &&edge) {
  edges_.push_back(std::move(edge));
}

void Sequence::PutRegion(std::shared_ptr<Region> &&region) {
  regions_.push_back(std::move(region));
}

size_t Sequence::GetIndex(const std::string &name) const {
  for (size_t i = 0; i < nodes_.size(); ++i) {
    std::shared_ptr<Node> node = nodes_[i];
    std::shared_ptr<Node> innerNode = std::dynamic_pointer_cast<Node>(node);
    if (innerNode && innerNode->GetName() == name) {
      return i;
    }
  }
#ifdef DEBUG
  assert(false && "unreachable");
#else
  __builtin_unreachable();
#endif
}

const std::vector<std::shared_ptr<Node>> &Sequence::GetNodes() const {
  return nodes_;
}

const std::vector<std::shared_ptr<Edge>> &Sequence::GetEdges() const {
  return edges_;
}

const std::vector<std::shared_ptr<Region>> &Sequence::GetRegions() const {
  return regions_;
}

} // namespace flow
} // namespace fluidml
