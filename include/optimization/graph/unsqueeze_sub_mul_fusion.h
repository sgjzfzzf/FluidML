#ifndef FLUIDML_OPTIMIZATION_GRAPH_UNSQUEEZE_SUB_MUL_H_
#define FLUIDML_OPTIMIZATION_GRAPH_UNSQUEEZE_SUB_MUL_H_

#include "optimization/graph/pass.h"

namespace fluidml {
namespace optimization {

class UnsqueezeSubMulPass : public GraphPass {
public:
  UnsqueezeSubMulPass() = default;
  UnsqueezeSubMulPass(const UnsqueezeSubMulPass &) = default;
  UnsqueezeSubMulPass(UnsqueezeSubMulPass &&) = default;
  static std::shared_ptr<UnsqueezeSubMulPass> Make();
  void Run(fluidml::graph::Node &node) const override;
};

} // namespace optimization
} // namespace fluidml

#endif
