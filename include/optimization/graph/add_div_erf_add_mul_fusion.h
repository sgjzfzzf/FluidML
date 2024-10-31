#ifndef FLUIDML_OPTIMIZATION_GRAPH_ADD_DIV_ERF_ADD_MUL_FUSION_H_
#define FLUIDML_OPTIMIZATION_GRAPH_ADD_DIV_ERF_ADD_MUL_FUSION_H_

#include "optimization/graph/pass.h"

namespace fluidml {
namespace optimization {
class AddDivErfAddMulFusionPass : public GraphPass {
public:
  AddDivErfAddMulFusionPass() = default;
  AddDivErfAddMulFusionPass(const AddDivErfAddMulFusionPass &) = default;
  AddDivErfAddMulFusionPass(AddDivErfAddMulFusionPass &&) = default;
  static std::shared_ptr<AddDivErfAddMulFusionPass> Make();
  void Run(fluidml::graph::Node &node) const override;
};
} // namespace optimization
} // namespace fluidml

#endif
