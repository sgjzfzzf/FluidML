#include "optimization/graph/manager.h"
#include "optimization/graph/gather_add_fusion.h"
#include "optimization/graph/unsqueeze_sub_mul_fusion.h"
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace optimization {

GraphPassesManager::GraphPassesManager(
    std::initializer_list<std::shared_ptr<GraphPass>> passes)
    : passes_(passes) {}

void GraphPassesManager::Run(graph::Graph &graph) const {
  std::vector<std::shared_ptr<graph::Node>> nodes = graph.GetAllNodes();
  for (std::shared_ptr<graph::Node> node : nodes) {
    for (std::shared_ptr<GraphPass> pass : passes_) {
#ifdef DEBUG
      assert(pass != nullptr);
      assert(node != nullptr);
      assert(graph.Check());
#endif
      pass->Run(*node);
    }
  }
}

void GraphPassesManager::RegisterAllPasses() {
  passes_.emplace_back(GatherAddFusionPass::Make());
  passes_.emplace_back(UnsqueezeSubMulPass::Make());
}

} // namespace optimization
} // namespace cpu_transformers
