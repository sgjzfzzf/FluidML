#include "optimization/graph/add_div_erf_add_mul_fusion.h"
#include "fmt/core.h"
#include "structure/graph/edge.h"
#include "structure/graph/graph.h"
#include "utils/isa.hpp"
#include <memory>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace optimization {

std::shared_ptr<AddDivErfAddMulFusionPass> AddDivErfAddMulFusionPass::Make() {
  return std::make_shared<AddDivErfAddMulFusionPass>();
}

void AddDivErfAddMulFusionPass::Run(cpu_transformers::graph::Node &node) const {
  graph::Graph *graph = node.GetGraph();
  if (graph == nullptr) {
    return;
  }
  if (node.GetOp() != graph::Node::Op::Add) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> add0_input_edges =
      node.GetInputEdges();
  if (add0_input_edges.size() != 2) {
    return;
  }
  std::shared_ptr<graph::Edge> add0_weight = add0_input_edges[0];
  std::shared_ptr<graph::Edge> add0_input = add0_input_edges[1];
#ifdef DEBUG
  assert(add0_weight != nullptr);
  assert(add0_input != nullptr);
#endif
  if (isa<graph::PureEdge>(add0_weight) &&
      isa<graph::ConstantTensorEdge>(add0_input)) {
    std::swap(add0_input, add0_weight);
  }
  if (!isa<graph::ConstantTensorEdge>(add0_weight) ||
      !isa<graph::PureEdge>(add0_input)) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> add0_output_edges =
      node.GetOutputEdges();
  if (add0_output_edges.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Edge> add0_output_edge = add0_output_edges[0];
#ifdef DEBUG
  assert(add0_output_edge != nullptr);
#endif
  std::vector<std::shared_ptr<graph::Node>> add0_output_nodes =
      add0_output_edge->GetOutputNodes();
  if (add0_output_nodes.size() != 2) {
    return;
  }
  std::shared_ptr<graph::Node> mul0_node = add0_output_nodes[0];
  std::shared_ptr<graph::Node> div_node = add0_output_nodes[1];
#ifdef DEBUG
  assert(mul0_node != nullptr);
  assert(div_node != nullptr);
#endif
  if (mul0_node->GetOp() == graph::Node::Op::Div &&
      div_node->GetOp() == graph::Node::Op::Mul) {
    std::swap(mul0_node, div_node);
  }
  if (mul0_node->GetOp() != graph::Node::Op::Mul ||
      div_node->GetOp() != graph::Node::Op::Div) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> div_input_edges =
      div_node->GetInputEdges();
  if (div_input_edges.size() != 2) {
    return;
  }
  std::shared_ptr<graph::Edge> div_input = div_input_edges[0];
  std::shared_ptr<graph::Edge> div_weight = div_input_edges[1];
#ifdef DEBUG
  assert(div_input != nullptr);
  assert(div_weight != nullptr);
#endif
  if (!isa<graph::PureEdge>(div_input) ||
      !isa<graph::ConstantScalarEdge>(div_weight)) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> div_output_edges =
      div_node->GetOutputEdges();
  if (div_output_edges.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Edge> div_output_edge = div_output_edges[0];
#ifdef DEBUG
  assert(div_output_edge != nullptr);
#endif
  std::vector<std::shared_ptr<graph::Node>> div_output_nodes =
      div_output_edge->GetOutputNodes();
  if (div_output_nodes.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Node> erf_node = div_output_nodes[0];
#ifdef DEBUG
  assert(erf_node != nullptr);
#endif
  if (erf_node->GetOp() != graph::Node::Op::Erf) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> erf_output_edges =
      erf_node->GetOutputEdges();
  if (erf_output_edges.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Edge> erf_output_edge = erf_output_edges[0];
#ifdef DEBUG
  assert(erf_output_edge != nullptr);
#endif
  std::vector<std::shared_ptr<graph::Node>> erf_output_nodes =
      erf_output_edge->GetOutputNodes();
  if (erf_output_nodes.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Node> add1_node = erf_output_nodes[0];
#ifdef DEBUG
  assert(add1_node != nullptr);
#endif
  if (add1_node->GetOp() != graph::Node::Op::Add) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> add1_input_edges =
      add1_node->GetInputEdges();
  if (add1_input_edges.size() != 2) {
    return;
  }
  std::shared_ptr<graph::Edge> add1_input = add1_input_edges[0];
  std::shared_ptr<graph::Edge> add1_weight = add1_input_edges[1];
#ifdef DEBUG
  assert(add1_input != nullptr);
  assert(add1_weight != nullptr);
#endif
  if (isa<graph::PureEdge>(add1_weight) &&
      isa<graph::ConstantScalarEdge>(add1_input)) {
    std::swap(add1_input, add1_weight);
  }
  if (!isa<graph::PureEdge>(add1_input) ||
      !isa<graph::ConstantScalarEdge>(add1_weight)) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> add1_output_edges =
      add1_node->GetOutputEdges();
  if (add1_output_edges.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Edge> add1_output_edge = add1_output_edges[0];
#ifdef DEBUG
  assert(add1_output_edge != nullptr);
#endif
  std::vector<std::shared_ptr<graph::Node>> add1_output_nodes =
      add1_output_edge->GetOutputNodes();
  if (add1_output_nodes.size() != 1 || add1_output_nodes[0] != mul0_node) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> mul0_output_edges =
      mul0_node->GetOutputEdges();
  if (mul0_output_edges.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Edge> mul0_output_edge = mul0_output_edges[0];
#ifdef DEBUG
  assert(mul0_output_edge != nullptr);
#endif
  std::vector<std::shared_ptr<graph::Node>> mul0_output_nodes =
      mul0_output_edge->GetOutputNodes();
  if (mul0_output_nodes.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Node> mul1_node = mul0_output_nodes[0];
#ifdef DEBUG
  assert(mul1_node != nullptr);
#endif
  if (mul1_node->GetOp() != graph::Node::Op::Mul) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> mul1_input_edges =
      mul1_node->GetInputEdges();
  if (mul1_input_edges.size() != 2) {
    return;
  }
  std::shared_ptr<graph::Edge> mul1_input = mul1_input_edges[0];
  std::shared_ptr<graph::Edge> mul1_weight = mul1_input_edges[1];
#ifdef DEBUG
  assert(mul1_input != nullptr);
  assert(mul1_weight != nullptr);
#endif
  if (isa<graph::PureEdge>(mul1_weight) &&
      isa<graph::ConstantScalarEdge>(mul1_input)) {
    std::swap(mul1_input, mul1_weight);
  }
  if (!isa<graph::PureEdge>(mul1_input) ||
      !isa<graph::ConstantScalarEdge>(mul1_weight)) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> mul1_output_edges =
      mul1_node->GetOutputEdges();
  if (mul1_output_edges.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Edge> mul1_output_edge = mul1_output_edges[0];
#ifdef DEBUG
  assert(mul1_output_edge != nullptr);
#endif
  std::string add_div_erf_add_mul_mul_node_name =
      fmt::format("{}-{}-{}-{}-{}-{}", node.GetName(), div_node->GetName(),
                  erf_node->GetName(), add1_node->GetName(),
                  mul0_node->GetName(), mul1_node->GetName());
  std::shared_ptr<graph::Node> add_div_erf_add_mul_mul_node =
      std::make_shared<graph::Node>(
          std::move(add_div_erf_add_mul_mul_node_name),
          graph::Node::Op::AddDivErfAddMulMul);
  std::shared_ptr<graph::Node> holder = add_div_erf_add_mul_mul_node;
  node.Delete();
  div_node->Delete();
  erf_node->Delete();
  add1_node->Delete();
  mul0_node->Delete();
  mul1_node->Delete();
  add0_weight->ClearOutput(node);
  add0_output_edge->Delete();
  div_weight->ClearOutput(*div_node);
  div_output_edge->Delete();
  erf_output_edge->Delete();
  add1_weight->ClearOutput(*add1_node);
  add1_output_edge->Delete();
  mul0_output_edge->Delete();
  mul1_weight->ClearOutput(*mul1_node);
  graph->PutNode(std::move(add_div_erf_add_mul_mul_node));
  add0_weight->PutOutput(*holder);
  div_weight->PutOutput(*holder);
  add1_weight->PutOutput(*holder);
  mul1_weight->PutOutput(*holder);
  add0_input->PutOutput(*holder);
  mul1_output_edge->PutInput(*holder);
#ifdef DEBUG
  assert(graph->Check());
#endif
}

} // namespace optimization
} // namespace cpu_transformers
