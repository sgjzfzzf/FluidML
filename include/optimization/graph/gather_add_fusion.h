#ifndef CPU_TRANSFORMERS_OPTIMIZATION_GRAPH_GATHER_ADD_FUSION_H_
#define CPU_TRANSFORMERS_OPTIMIZATION_GRAPH_GATHER_ADD_FUSION_H_

#include "optimization/graph/pass.h"

namespace cpu_transformers {
namespace optimization {

class GatherAddFusionPass : public GraphPass {
public:
  GatherAddFusionPass() = default;
  GatherAddFusionPass(const GatherAddFusionPass &) = default;
  GatherAddFusionPass(GatherAddFusionPass &&) = default;
  void Run(cpu_transformers::graph::Node &node) const override;
};

} // namespace optimization
} // namespace cpu_transformers

#endif
