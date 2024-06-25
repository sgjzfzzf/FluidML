#ifndef CPU_TRANSFORMERS_OPTIMIZATION_GRAPH_UNSQUEEZE_SUB_MUL_H_
#define CPU_TRANSFORMERS_OPTIMIZATION_GRAPH_UNSQUEEZE_SUB_MUL_H_

#include "optimization/graph/pass.h"

namespace cpu_transformers {
namespace optimization {

class UnsqueezeSubMulPass : public GraphPass {
public:
  UnsqueezeSubMulPass() = default;
  UnsqueezeSubMulPass(const UnsqueezeSubMulPass &) = default;
  UnsqueezeSubMulPass(UnsqueezeSubMulPass &&) = default;
  static std::shared_ptr<UnsqueezeSubMulPass> Make();
  void Run(cpu_transformers::graph::Node &node) const override;
};

} // namespace optimization
} // namespace cpu_transformers

#endif
