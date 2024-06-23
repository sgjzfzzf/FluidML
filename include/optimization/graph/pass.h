#ifndef CPU_TRANSFORMERS_OPTIMIZATION_GRAPH_PASS_H_
#define CPU_TRANSFORMERS_OPTIMIZATION_GRAPH_PASS_H_

#include "optimization/pass.h"
#include "structure/graph/node.h"

namespace cpu_transformers {
namespace optimization {

class GraphPass : public Pass {
public:
  GraphPass() = default;
  GraphPass(const GraphPass &) = default;
  GraphPass(GraphPass &&) = default;
  virtual void Run(cpu_transformers::graph::Node &node) const = 0;
};

} // namespace optimization
} // namespace cpu_transformers

#endif
