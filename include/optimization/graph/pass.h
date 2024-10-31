#ifndef FLUIDML_OPTIMIZATION_GRAPH_PASS_H_
#define FLUIDML_OPTIMIZATION_GRAPH_PASS_H_

#include "optimization/pass.h"
#include "structure/graph/node.h"

namespace fluidml {
namespace optimization {

class GraphPass : public Pass {
public:
  GraphPass() = default;
  GraphPass(const GraphPass &) = default;
  GraphPass(GraphPass &&) = default;
  virtual void Run(fluidml::graph::Node &node) const = 0;
};

} // namespace optimization
} // namespace fluidml

#endif
