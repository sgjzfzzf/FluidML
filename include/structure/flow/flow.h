#ifndef CPU_TRANSFORMERS_STRUCTURE_FLOW_FLOW_H_
#define CPU_TRANSFORMERS_STRUCTURE_FLOW_FLOW_H_

#include "structure/flow/fwd.h"
#include <memory>
#include <string>
#include <unordered_map>
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
  std::shared_ptr<Region> GetRegion(const std::string &name) const;
  std::vector<std::shared_ptr<Node>> GetNodes() const;
  std::vector<std::shared_ptr<Edge>> GetEdges(const std::string &name) const;
  std::vector<std::shared_ptr<Edge>> GetEdges() const;
  std::vector<std::shared_ptr<Region>> GetRegions() const;
  std::shared_ptr<OwnToEdge> GetInputEdge(const SingleInputNode &node) const;
  std::shared_ptr<OwnFromEdge> GetOutputEdge(const SingleInputNode &node) const;
  std::shared_ptr<OwnToEdge> GetLhsEdge(const DoubleInputsNode &node) const;
  std::shared_ptr<OwnToEdge> GetRhsEdge(const DoubleInputsNode &node) const;
  std::shared_ptr<OwnFromEdge>
  GetOutputEdge(const DoubleInputsNode &node) const;
#ifdef DEBUG
  bool Check() const;
  bool IsNoOverlapFlow() const;
#endif

private:
  std::unordered_map<std::string, std::shared_ptr<Node>> nodes_;
  std::unordered_multimap<std::string, std::shared_ptr<Edge>> edges_;
  std::unordered_map<std::string, std::shared_ptr<Region>> regions_;
};

} // namespace flow
} // namespace cpu_transformers

#endif
