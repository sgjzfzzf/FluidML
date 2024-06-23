#include "optimization/graph/gather_add_fusion.h"
#include "fmt/core.h"
#include "structure/graph/edge.h"
#include "structure/graph/graph.h"
#include "structure/graph/node.h"
#include <memory>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace optimization {

void GatherAddFusionPass::Run(cpu_transformers::graph::Node &node) const {
  graph::Graph *graph = node.GetGraph();
  if (graph == nullptr) {
    return;
  }
  if (node.GetOp() != graph::Node::Op::Gather) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> input0_edges = node.GetInputEdges();
  if (input0_edges.size() != 2) {
    return;
  }
  std::shared_ptr<graph::Edge> gather_data_edge = input0_edges[0];
#ifdef DEBUG
  assert(gather_data_edge != nullptr);
#endif
  std::shared_ptr<graph::Edge> input_edge = input0_edges[1];
  std::vector<std::shared_ptr<graph::Edge>> output0_edges =
      node.GetOutputEdges();
  if (output0_edges.size() != 1) {
    return;
  }
  std::shared_ptr<graph::PureEdge> add0_edge =
      std::dynamic_pointer_cast<graph::PureEdge>(output0_edges[0]);
  if (add0_edge == nullptr) {
    return;
  }
  std::vector<std::shared_ptr<graph::Node>> output0_nodes =
      add0_edge->GetOutputNodes();
  if (output0_nodes.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Node> add0_node = output0_nodes[0];
  if (add0_node->GetOp() != graph::Node::Op::Add) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> input1_edges =
      add0_node->GetInputEdges();
  if (input1_edges.size() != 2) {
    return;
  }
#ifdef DEBUG
  std::shared_ptr<graph::Edge> add0_input_edge = input1_edges[0];
#endif
  std::shared_ptr<graph::Edge> add0_weight_edge =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(input1_edges[1]);
#ifdef DEBUG
  assert(add0_input_edge == add0_edge);
  assert(add0_weight_edge != nullptr);
#endif
  std::vector<std::shared_ptr<graph::Edge>> output1_edges =
      add0_node->GetOutputEdges();
  if (output1_edges.size() != 1) {
    return;
  }
  std::shared_ptr<graph::PureEdge> add1_edge =
      std::dynamic_pointer_cast<graph::PureEdge>(output1_edges[0]);
  if (add1_edge == nullptr) {
    return;
  }
  std::vector<std::shared_ptr<graph::Node>> output1_nodes =
      add1_edge->GetOutputNodes();
  if (output1_nodes.size() != 1) {
    return;
  }
  std::shared_ptr<graph::Node> add1_node = output1_nodes[0];
  if (add1_node->GetOp() != graph::Node::Op::Add) {
    return;
  }
  std::vector<std::shared_ptr<graph::Edge>> input2_edges =
      add1_node->GetInputEdges();
  if (input2_edges.size() != 2) {
    return;
  }
#ifdef DEBUG
  std::shared_ptr<graph::Edge> add1_input_edge = input2_edges[0];
#endif
  std::shared_ptr<graph::Edge> add1_weight_edge =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(input2_edges[1]);
#ifdef DEBUG
  assert(add1_input_edge == add1_edge);
  assert(add1_weight_edge != nullptr);
#endif
  std::vector<std::shared_ptr<graph::Edge>> output2_edges =
      add1_node->GetOutputEdges();
  if (output2_edges.size() != 1) {
    return;
  }
  std::shared_ptr<graph::PureEdge> output_edge =
      std::dynamic_pointer_cast<graph::PureEdge>(output2_edges[0]);
  std::string gather_add_add_name = fmt::format(
      "{}-{}-{}", node.GetName(), add0_node->GetName(), add1_node->GetName());
  std::shared_ptr<graph::Node> gather_add_add_node =
      std::make_shared<graph::Node>(std::move(gather_add_add_name),
                                    graph::Node::Op::GatherAddAdd);
  std::shared_ptr<graph::Node> holder = gather_add_add_node;
  add0_edge->Delete();
  add1_edge->Delete();
  gather_data_edge->ClearOutputs();
  add0_weight_edge->ClearOutputs();
  add1_weight_edge->ClearOutputs();
  node.Delete();
  add0_node->Delete();
  add1_node->Delete();
  input_edge->ClearOutputs();
  output_edge->ClearInputs();
  graph->PutNode(std::move(gather_add_add_node));
  gather_data_edge->PutOutput(*holder);
  add0_weight_edge->PutOutput(*holder);
  add1_weight_edge->PutOutput(*holder);
  input_edge->PutOutput(*holder);
  output_edge->PutInput(*holder);
#ifdef DEBUG
  assert(graph->Check());
#endif
}

} // namespace optimization
} // namespace cpu_transformers
