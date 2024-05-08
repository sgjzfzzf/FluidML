#ifndef CPU_TRANSFORMERS_STRUCTURE_FLOW_FLOW_H_
#define CPU_TRANSFORMERS_STRUCTURE_FLOW_FLOW_H_

#include "structure/flow/edge.h"
#include "structure/flow/node.h"
#include "structure/flow/region.h"
#include <memory>
#include <vector>

namespace cpu_transformers {
namespace flow {
class Flow {
public:
  Flow() = default;
  Flow(const Flow &flow) = delete;
  Flow(Flow &&flow) = default;
  void PutNode(std::shared_ptr<Node> &&node);
  void PutEdge(std::shared_ptr<Edge> &&edge);
  void PutRegion(std::shared_ptr<Region> &&region);
  std::shared_ptr<Node> GetNode(const std::string &name) const;
  std::shared_ptr<Edge> GetEdge(const std::string &name) const;
  const std::vector<std::shared_ptr<Node>> &GetNodes() const;
  const std::vector<std::shared_ptr<Edge>> &GetEdges() const;
  const std::vector<std::shared_ptr<Region>> &GetRegions() const;

private:
  std::vector<std::shared_ptr<Node>> nodes_;
  std::vector<std::shared_ptr<Edge>> edges_;
  std::vector<std::shared_ptr<Region>> regions_;
};
} // namespace flow
} // namespace cpu_transformers

#endif
