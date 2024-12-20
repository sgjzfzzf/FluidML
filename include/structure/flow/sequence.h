#ifndef FLUIDML_STRUCTURE_FLOW_SEQUENCE_H_
#define FLUIDML_STRUCTURE_FLOW_SEQUENCE_H_

#include "structure/flow/fwd.h"
#include <memory>
#include <vector>

namespace fluidml {
namespace flow {

class Sequence {
public:
  Sequence() = default;
  Sequence(const Sequence &sequence) = delete;
  Sequence(Sequence &&sequence) = default;
  void PutNode(std::shared_ptr<Node> &&node);
  void PutEdge(std::shared_ptr<Edge> &&edge);
  void PutRegion(std::shared_ptr<Region> &&region);
  size_t GetIndex(const std::string &name) const;
  const std::vector<std::shared_ptr<Node>> &GetNodes() const;
  const std::vector<std::shared_ptr<Edge>> &GetEdges() const;
  const std::vector<std::shared_ptr<Region>> &GetRegions() const;

private:
  std::vector<std::shared_ptr<Node>> nodes_;
  std::vector<std::shared_ptr<Edge>> edges_;
  std::vector<std::shared_ptr<Region>> regions_;
};

} // namespace flow
} // namespace fluidml

#endif
