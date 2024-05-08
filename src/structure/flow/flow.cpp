#include "structure/flow/flow.h"
#include "structure/flow/node.h"
#include <memory>
#include <string>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace flow {

void Flow::PutNode(std::shared_ptr<Node> &&node) {
#ifdef DEBUG
  assert(node != nullptr);
  const std::string &name = node->GetName();
  for (const std::shared_ptr<Node> &n : nodes_) {
    assert(n->GetName() != name);
  }
#endif
  nodes_.push_back(std::move(node));
}

void Flow::PutEdge(std::shared_ptr<Edge> &&edge) {
#ifdef DEBUG
  assert(edge != nullptr);
  if (std::shared_ptr<MemoryEdge> memory_edge0 =
          std::dynamic_pointer_cast<MemoryEdge>(edge)) {
    for (const std::shared_ptr<Edge> &edge1 : edges_) {
      if (std::shared_ptr<MemoryEdge> memory_edge1 =
              std::dynamic_pointer_cast<MemoryEdge>(edge1)) {
        assert(memory_edge1->GetFrom() != memory_edge0->GetFrom() ||
               memory_edge1->GetTo() != memory_edge0->GetTo());
      }
    }
  }
#endif
  edges_.push_back(std::move(edge));
}

void Flow::PutRegion(std::shared_ptr<Region> &&region) {
#ifdef DEBUG
  assert(region != nullptr);
  const std::string &name = region->GetName();
  for (const std::shared_ptr<Region> &r : regions_) {
    assert(r->GetName() != name);
  }
#endif
  regions_.push_back(std::move(region));
}

std::shared_ptr<Node> Flow::GetNode(const std::string &name) const {
  for (const std::shared_ptr<Node> &node : nodes_) {
    if (node->GetName() == name) {
      return node;
    }
  }
  return nullptr;
}

std::shared_ptr<Edge> Flow::GetEdge(const std::string &name) const {
  for (const std::shared_ptr<Edge> &edge : edges_) {
    if (edge->GetName() == name) {
      return edge;
    }
  }
  return nullptr;
}

const std::vector<std::shared_ptr<Node>> &Flow::GetNodes() const {
  return nodes_;
}

const std::vector<std::shared_ptr<Edge>> &Flow::GetEdges() const {
  return edges_;
}

const std::vector<std::shared_ptr<Region>> &Flow::GetRegions() const {
  return regions_;
}

} // namespace flow
} // namespace cpu_transformers
