#include "optimization/graph/unsqueeze_sub_mul_fusion.h"
#include "fmt/core.h"
#include "structure/graph/edge.h"
#include "structure/graph/graph.h"
#include <memory>
#ifdef DEBUG
#include <cassert>
#endif

namespace fluidml {
namespace optimization {

std::shared_ptr<UnsqueezeSubMulPass> UnsqueezeSubMulPass::Make() {
  return std::make_shared<UnsqueezeSubMulPass>();
}

void UnsqueezeSubMulPass::Run(fluidml::graph::Node &node) const {
  graph::Graph *graph = node.GetGraph();
  if (graph == nullptr) {
    return;
  }
  if (node.GetOp() != graph::Node::Op::Unsqueeze) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> unsqueeze_input_edges =
      node.GetInputEdges();
  if (unsqueeze_input_edges.size() != 2) {
    return;
  }
  std::shared_ptr<graph::Edge> input_edge = unsqueeze_input_edges[0];
  std::shared_ptr<graph::ConstantTensorEdge> unsqueeze_axes_edge =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(
          unsqueeze_input_edges[1]);
#ifdef DEBUG
  assert(input_edge != nullptr);
#endif
  if (unsqueeze_axes_edge == nullptr) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> unsqueeze_output_edges =
      node.GetOutputEdges();
  if (unsqueeze_output_edges.size() != 1) {
    return;
  }
  std::shared_ptr<graph::PureEdge> unsqueeze_output_edge =
      std::dynamic_pointer_cast<graph::PureEdge>(unsqueeze_output_edges[0]);
  if (unsqueeze_output_edge == nullptr) {
    return;
  }
  std::vector<std::shared_ptr<graph::Node>> unsqueeze_output_nodes =
      unsqueeze_output_edge->GetOutputNodes();
  if (unsqueeze_output_nodes.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Node> sub_node = unsqueeze_output_nodes[0];
#ifdef DEBUG
  assert(sub_node != nullptr);
#endif
  if (sub_node->GetOp() != graph::Node::Op::Sub) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> sub_input_edges =
      sub_node->GetInputEdges();
  if (sub_input_edges.size() != 2) {
    return;
  }
  std::shared_ptr<graph::ConstantScalarEdge> sub_val_edge =
      std::dynamic_pointer_cast<graph::ConstantScalarEdge>(sub_input_edges[0]);
  std::shared_ptr<graph::PureEdge> sub_input_edge =
      std::dynamic_pointer_cast<graph::PureEdge>(sub_input_edges[1]);
  if (sub_val_edge == nullptr || sub_input_edge == nullptr) {
    return;
  }
#ifdef DEBUG
  assert(unsqueeze_output_edge == sub_input_edge);
#endif
  std::vector<std::shared_ptr<graph::Edge>> sub_output_edges =
      sub_node->GetOutputEdges();
  if (sub_output_edges.size() != 1) {
    return;
  }
  std::shared_ptr<graph::PureEdge> sub_output_edge =
      std::dynamic_pointer_cast<graph::PureEdge>(sub_output_edges[0]);
  if (sub_output_edge == nullptr) {
    return;
  }
  std::vector<std::shared_ptr<graph::Node>> sub_nodes =
      sub_output_edge->GetOutputNodes();
  if (sub_nodes.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Node> mul_node = sub_nodes[0];
#ifdef DEBUG
  assert(mul_node != nullptr);
#endif
  if (mul_node->GetOp() != graph::Node::Op::Mul) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> mul_input_edges =
      mul_node->GetInputEdges();
  if (mul_input_edges.size() != 2) {
    return;
  }
  std::shared_ptr<graph::PureEdge> mul_input_edge =
      std::dynamic_pointer_cast<graph::PureEdge>(mul_input_edges[0]);
  std::shared_ptr<graph::ConstantScalarEdge> mul_weight_edge =
      std::dynamic_pointer_cast<graph::ConstantScalarEdge>(mul_input_edges[1]);
  if (mul_input_edge == nullptr || mul_weight_edge == nullptr) {
    return;
  }
#ifdef DEBUG
  assert(sub_output_edge == mul_input_edge);
#endif
  std::vector<std::shared_ptr<graph::Edge>> mul_output_edges =
      mul_node->GetOutputEdges();
  if (mul_output_edges.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Edge> output_edge = mul_output_edges[0];
#ifdef DEBUG
  assert(output_edge != nullptr);
#endif
  std::string name = fmt::format("{}-{}-{}", node.GetName(),
                                 sub_node->GetName(), mul_node->GetName());
  std::shared_ptr<graph::Node> unsqueeze_sub_mul_node =
      std::make_shared<graph::Node>(std::move(name),
                                    graph::Node::Op::UnsqueezeSubMul);
  std::shared_ptr<graph::Node> holder = unsqueeze_sub_mul_node;
  sub_input_edge->Delete();
  mul_input_edge->Delete();
  node.Delete();
  sub_node->Delete();
  mul_node->Delete();
  graph->PutNode(std::move(unsqueeze_sub_mul_node));
  unsqueeze_axes_edge->ClearOutput(node);
  unsqueeze_axes_edge->PutOutput(*holder);
  sub_val_edge->ClearOutput(*sub_node);
  sub_val_edge->PutOutput(*holder);
  mul_weight_edge->ClearOutput(*mul_node);
  mul_weight_edge->PutOutput(*holder);
  input_edge->ClearOutput(node);
  input_edge->PutOutput(*holder);
  output_edge->ClearInput(*mul_node);
  output_edge->PutInput(*holder);
#ifdef DEBUG
  assert(graph->Check());
#endif
}

} // namespace optimization
} // namespace fluidml
