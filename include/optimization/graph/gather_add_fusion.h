#ifndef FLUIDML_OPTIMIZATION_GRAPH_GATHER_ADD_FUSION_H_
#define FLUIDML_OPTIMIZATION_GRAPH_GATHER_ADD_FUSION_H_

#include "optimization/graph/pass.h"

namespace fluidml {
namespace optimization {

class GatherAddFusionPass : public GraphPass {
public:
  GatherAddFusionPass() = default;
  GatherAddFusionPass(const GatherAddFusionPass &) = default;
  GatherAddFusionPass(GatherAddFusionPass &&) = default;
  static std::shared_ptr<GatherAddFusionPass> Make();
  void Run(fluidml::graph::Node &node) const override;
};

} // namespace optimization
} // namespace fluidml

#endif
