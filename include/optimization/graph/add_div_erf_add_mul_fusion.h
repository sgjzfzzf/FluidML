#ifndef CPU_TRANSFORMERS_OPTIMIZATION_GRAPH_ADD_DIV_ERF_ADD_MUL_FUSION_H_
#define CPU_TRANSFORMERS_OPTIMIZATION_GRAPH_ADD_DIV_ERF_ADD_MUL_FUSION_H_

#include "optimization/graph/pass.h"

namespace cpu_transformers {
namespace optimization {
class AddDivErfAddMulFusionPass : public GraphPass {
public:
  AddDivErfAddMulFusionPass() = default;
  AddDivErfAddMulFusionPass(const AddDivErfAddMulFusionPass &) = default;
  AddDivErfAddMulFusionPass(AddDivErfAddMulFusionPass &&) = default;
  static std::shared_ptr<AddDivErfAddMulFusionPass> Make();
  void Run(cpu_transformers::graph::Node &node) const override;
};
} // namespace optimization
} // namespace cpu_transformers

#endif
