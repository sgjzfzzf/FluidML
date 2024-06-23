#include "worker/converter.h"
#include "structure/flow/edge.h"
#include "structure/flow/node.h"
#include "structure/flow/region.h"
#include "structure/graph/edge.h"
#include "structure/graph/node.h"
#include "utils/isa.hpp"
#include <cmath>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#ifdef DEBUG
#include "exception/unimplemented_exception.h"
#include "exception/unreachable_exception.h"
#include <cassert>
#endif

namespace cpu_transformers {
namespace worker {
flow::Flow Converter::Run(const graph::Graph &graph) {
  flow::Flow flow;
  for (std::shared_ptr<graph::Edge> edge : graph.GetAllEdges()) {
    if (std::shared_ptr<graph::InputEdge> input_edge =
            std::dynamic_pointer_cast<graph::InputEdge>(edge)) {
      std::string name = input_edge->GetName();
      std::vector<std::shared_ptr<graph::Node>> tos =
          graph.GetEdgeTo(*input_edge);
#ifdef DEBUG
      assert(tos.size() == 1);
#endif
      std::shared_ptr<graph::Node> to = tos[0];
      std::string to_name = to->GetName();
      Meta meta = input_edge->GetMeta();
      name = input_edge->GetName();
      std::string from_name = name;
      std::shared_ptr<flow::InputRegion> region =
          std::make_shared<flow::InputRegion>(std::move(name), std::move(meta));
      std::shared_ptr<flow::InputRegion> region_clone = region;
      std::shared_ptr<flow::InputEdge> ptr = std::make_shared<flow::InputEdge>(
          std::move(region_clone), std::move(to_name));
      flow.PutEdge(std::move(ptr));
      flow.PutRegion(std::move(region));
    } else if (std::shared_ptr<graph::OutputEdge> output_edge =
                   std::dynamic_pointer_cast<graph::OutputEdge>(edge)) {
      std::string name = output_edge->GetName();
      std::shared_ptr<graph::Node> from = graph.GetEdgeFrom(*output_edge);
      std::string from_name = from->GetName();
      Meta meta = output_edge->GetMeta();
      name = output_edge->GetName();
      std::string to_name = name;
      std::shared_ptr<flow::OutputRegion> region =
          std::make_shared<flow::OutputRegion>(std::move(name),
                                               std::move(meta));
      std::shared_ptr<flow::OutputRegion> region_clone = region;
      std::shared_ptr<flow::OutputEdge> output_ptr =
          std::make_shared<flow::OutputEdge>(std::move(region_clone),
                                             std::move(from_name));
      flow.PutEdge(std::move(output_ptr));
      region_clone = region;
      flow.PutRegion(std::move(region_clone));
      std::vector<std::shared_ptr<graph::Node>> tos =
          graph.GetEdgeTo(*output_edge);
      for (std::shared_ptr<graph::Node> to : tos) {
        from_name = from->GetName();
        to_name = to->GetName();
        region_clone = region;
        std::shared_ptr<flow::MemoryEdge> memory_ptr =
            std::make_shared<flow::MemoryEdge>(std::move(region_clone),
                                               std::move(from_name),
                                               std::move(to_name));
        flow.PutEdge(std::move(memory_ptr));
      }
    } else if (std::shared_ptr<graph::PureEdge> pure_edge =
                   std::dynamic_pointer_cast<graph::PureEdge>(edge)) {
      std::string name = pure_edge->GetName();
      Meta meta = pure_edge->GetMeta();
      std::shared_ptr<flow::InnerRegion> region =
          std::make_shared<flow::InnerRegion>(std::move(name), std::move(meta));
      std::shared_ptr<flow::InnerRegion> region_clone = region;
      flow.PutRegion(std::move(region_clone));
      std::shared_ptr<graph::Node> from = graph.GetEdgeFrom(*pure_edge);
      std::vector<std::shared_ptr<graph::Node>> tos =
          graph.GetEdgeTo(*pure_edge);
      for (const std::shared_ptr<graph::Node> &to : tos) {
        std::string from_name = from->GetName();
        std::string to_name = to->GetName();
        region_clone = region;
        std::shared_ptr<flow::MemoryEdge> ptr =
            std::make_shared<flow::MemoryEdge>(std::move(region_clone),
                                               std::move(from_name),
                                               std::move(to_name));
        flow.PutEdge(std::move(ptr));
      }
    }
  }
  for (std::shared_ptr<graph::Node> node : graph.GetAllNodes()) {
    std::shared_ptr<flow::Node> ptr = nullptr;
    switch (node->GetOp()) {
    case graph::Node::Op::Add:
      convertAddNode(flow, graph, *node);
      break;
    case graph::Node::Op::Div:
      convertDivNode(flow, graph, *node);
      break;
    case graph::Node::Op::Erf:
      convertErfNode(flow, graph, *node);
      break;
    case graph::Node::Op::Gather:
      convertGatherNode(flow, graph, *node);
      break;
    case graph::Node::Op::GatherAddAdd:
      convertGatherAddAddNode(flow, graph, *node);
      break;
    case graph::Node::Op::Gemm:
      convertGemmNode(flow, graph, *node);
      break;
    case graph::Node::Op::LayerNormalization:
      convertLayerNormalizationNode(flow, graph, *node);
      break;
    case graph::Node::Op::MatMul:
      convertMatMulNode(flow, graph, *node);
      break;
    case graph::Node::Op::Mul:
      convertMulNode(flow, graph, *node);
      break;
    case graph::Node::Op::Pow:
      convertPowNode(flow, graph, *node);
      break;
    case graph::Node::Op::Reshape:
      convertReshapeNode(flow, graph, *node);
      break;
    case graph::Node::Op::Softmax:
      convertSoftmaxNode(flow, graph, *node);
      break;
    case graph::Node::Op::Split:
      convertSplitNode(flow, graph, *node);
      break;
    case graph::Node::Op::Sub:
      convertSubNode(flow, graph, *node);
      break;
    case graph::Node::Op::Tanh:
      convertTanhNode(flow, graph, *node);
      break;
    case graph::Node::Op::Transpose:
      convertTransposeNode(flow, graph, *node);
      break;
    case graph::Node::Op::Unsqueeze:
      convertUnsqueezeNode(flow, graph, *node);
      break;
    case graph::Node::Op::Where:
      convertWhereNode(flow, graph, *node);
      break;
    default:
#ifdef DEBUG
      throw UnimplementedException();
#else
      __builtin_unreachable();
#endif
    }
  }
  return flow;
}

void Converter::convertAddNode(flow::Flow &flow, const graph::Graph &graph,
                               const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Add);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input_lhs = inputs[0];
  std::shared_ptr<graph::Edge> input_rhs = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(input_lhs != nullptr);
  assert(input_rhs != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::AddNode> ptr = nullptr;
  std::shared_ptr<graph::ConstantEdge> input_lhs_as_constant =
      std::dynamic_pointer_cast<graph::ConstantEdge>(input_lhs);
  std::shared_ptr<graph::ConstantEdge> input_rhs_as_constant =
      std::dynamic_pointer_cast<graph::ConstantEdge>(input_rhs);
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(output_as_non_constant != nullptr);
#endif
  Meta output_meta = output_as_non_constant->GetMeta();
  if (input_lhs_as_constant == nullptr && input_rhs_as_constant == nullptr) {
    std::shared_ptr<graph::NonConstantEdge> input_lhs_as_non_constant =
        std::dynamic_pointer_cast<graph::NonConstantEdge>(input_lhs);
    std::shared_ptr<graph::NonConstantEdge> input_rhs_as_non_constant =
        std::dynamic_pointer_cast<graph::NonConstantEdge>(input_rhs);
#ifdef DEBUG
    assert(input_lhs_as_non_constant != nullptr);
    assert(input_rhs_as_non_constant != nullptr);
#endif
    const std::string &input_lhs_name = input_lhs_as_non_constant->GetName();
    const std::string &input_rhs_name = input_rhs_as_non_constant->GetName();
    const std::string &output_name = output->GetName();
    std::shared_ptr<flow::Edge> input_lhs_tr = flow.GetEdge(input_lhs_name);
    std::shared_ptr<flow::Edge> input_rhs_ptr = flow.GetEdge(input_rhs_name);
    std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
    assert(input_lhs_tr != nullptr);
    assert(input_rhs_ptr != nullptr);
    assert(output_ptr != nullptr);
    Meta input_lhs_meta = input_lhs_tr->GetMeta();
    Meta input_rhs_meta = input_rhs_ptr->GetMeta();
    std::optional<Meta> output_meta_opt =
        BroadcastShape(input_lhs_meta, input_rhs_meta, output_meta.GetType());
    assert(output_meta_opt.has_value());
    assert(*output_meta_opt == output_meta);
#endif
    ptr = std::make_shared<flow::AddCommonNode>(
        std::move(name), std::move(input_lhs_tr), std::move(input_rhs_ptr),
        std::move(output_ptr));
  }
#ifdef DEBUG
  else if (input_lhs_as_constant != nullptr &&
           input_rhs_as_constant != nullptr) {
    throw UnreachableException();
  }
#endif
  else {
    std::shared_ptr<graph::ConstantEdge> input_constant = nullptr;
    std::shared_ptr<graph::NonConstantEdge> input_non_constant = nullptr;
    if (input_lhs_as_constant != nullptr && input_rhs_as_constant == nullptr) {
      input_constant = input_lhs_as_constant;
      input_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input_rhs);
    } else if (input_lhs_as_constant == nullptr &&
               input_rhs_as_constant != nullptr) {
      input_constant = input_rhs_as_constant;
      input_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input_lhs);
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
#ifdef DEBUG
    assert(input_constant != nullptr);
    assert(input_non_constant != nullptr);
#endif
    if (std::shared_ptr<graph::ConstantScalarEdge> input_constant_scalar =
            std::dynamic_pointer_cast<graph::ConstantScalarEdge>(
                input_constant)) {
      const std::string &input_name = input_non_constant->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
      assert(input_ptr != nullptr);
      assert(output_ptr != nullptr);
#endif
      ptr = std::make_shared<flow::AddConstantScalarNode>(
          std::move(name), input_constant_scalar->GetType(),
          input_constant_scalar->GetValue(), std::move(input_ptr),
          std::move(output_ptr));
    } else if (std::shared_ptr<graph::ConstantTensorEdge>
                   input_constant_tensor =
                       std::dynamic_pointer_cast<graph::ConstantTensorEdge>(
                           input_constant)) {
      Tensor tensor = input_constant_tensor->GetValue();
      const std::string &input_name = input_non_constant->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
      assert(input_ptr != nullptr);
      assert(output_ptr != nullptr);
#endif
      ptr = std::make_shared<flow::AddConstantTensorNode>(
          std::move(name), std::move(tensor), std::move(input_ptr),
          std::move(output_ptr));
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  }
  flow.PutNode(std::move(ptr));
}

void Converter::convertDivNode(flow::Flow &flow, const graph::Graph &graph,
                               const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Div);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input_lhs = inputs[0];
  std::shared_ptr<graph::Edge> input_rhs = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(input_lhs != nullptr);
  assert(input_rhs != nullptr);
#endif
  std::shared_ptr<flow::DivNode> ptr = nullptr;
  // The support format of Div operator is limited to the following:
  // the left operator is a tensor, and it's a non-constant edge
  // the right operator is a scalar, and it's a constant scalar edge
  // If new formats occur, the code should be updated.
  std::shared_ptr<graph::ConstantScalarEdge> input_rhs_as_constant_scalar =
      std::dynamic_pointer_cast<graph::ConstantScalarEdge>(input_rhs);
#ifdef DEBUG
  assert(isa<graph::NonConstantEdge>(input_lhs));
  assert(input_rhs_as_constant_scalar != nullptr);
  assert(isa<graph::NonConstantEdge>(output));
#endif
  const std::string &input_name = input_lhs->GetName();
  const std::string &output_name = output->GetName();
  std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
  std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
  assert(input_ptr != nullptr);
  assert(output_ptr != nullptr);
  Meta meta = input_ptr->GetMeta();
  assert(meta == output_ptr->GetMeta());
#endif
  ptr = std::make_shared<flow::DivConstantScalarNode>(
      std::move(name), input_rhs_as_constant_scalar->GetType(),
      input_rhs_as_constant_scalar->GetValue(), std::move(input_ptr),
      std::move(output_ptr));
  flow.PutNode(std::move(ptr));
}

void Converter::convertErfNode(flow::Flow &flow, const graph::Graph &graph,
                               const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Erf);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::ErfNode> ptr = nullptr;
#ifdef DEBUG
  assert(isa<graph::NonConstantEdge>(input));
  assert(isa<graph::NonConstantEdge>(output));
#endif
  const std::string &input_name = input->GetName();
  const std::string &output_name = output->GetName();
  std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
  std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
  assert(input_ptr != nullptr);
  assert(output_ptr != nullptr);
  Meta input_meta = input_ptr->GetMeta();
  const Meta &output_meta = output_ptr->GetMeta();
  assert(input_meta == output_meta);
#endif
  ptr = std::make_shared<flow::ErfNode>(std::move(name), std::move(input_ptr),
                                        std::move(output_ptr));
  flow.PutNode(std::move(ptr));
}

// TODO: It only supports two formats of Gather operator:
// 1. GatherConstantIndexScalarNode: the first input is a tensor, and the second
// input is a scalar edge.
// 2. GatherConstantDataTensorNode: the first input is a scalar, and the second
// input is a tensor edge.
// If new formats occur, the code should be updated.
void Converter::convertGatherNode(flow::Flow &flow, const graph::Graph &graph,
                                  const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Gather);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input_lhs = inputs[0];
  std::shared_ptr<graph::Edge> input_rhs = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
  size_t axis = 0;
  if (node.HasAttribute(flow::GatherNode::kAxisAttrName)) {
    axis = node.GetAttribute(flow::GatherNode::kAxisAttrName).GetInt64();
  }
#ifdef DEBUG
  assert(input_lhs != nullptr);
  assert(input_rhs != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::GatherNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  if (std::shared_ptr<graph::NonConstantEdge> input_lhs_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input_lhs)) {
    if (std::shared_ptr<graph::ConstantScalarEdge>
            input_rhs_as_constant_scalar =
                std::dynamic_pointer_cast<graph::ConstantScalarEdge>(
                    input_rhs)) {
      const std::string &input_name = input_lhs_as_non_constant->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
      assert(input_ptr != nullptr);
      assert(output_ptr != nullptr);
#endif
      ptr = std::make_shared<flow::GatherConstantIndexScalarNode>(
          std::move(name), std::move(input_ptr),
          std::lround(input_rhs_as_constant_scalar->GetValue()),
          std::move(output_ptr), axis);
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  } else if (std::shared_ptr<graph::ConstantTensorEdge>
                 inputLhs_as_constant_tensor =
                     std::dynamic_pointer_cast<graph::ConstantTensorEdge>(
                         input_lhs)) {
    if (std::shared_ptr<graph::NonConstantEdge> input_rhs_as_non_constant =
            std::dynamic_pointer_cast<graph::NonConstantEdge>(input_rhs)) {
      Tensor tensor = inputLhs_as_constant_tensor->GetValue();
      const std::string &input_name = input_rhs_as_non_constant->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
      assert(input_ptr != nullptr);
      assert(output_ptr != nullptr);
#endif
      ptr = std::make_shared<flow::GatherConstantDataTensorNode>(
          std::move(name), std::move(tensor), std::move(input_ptr),
          std::move(output_ptr), axis);
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  } else {
#ifdef DEBUG
    throw UnreachableException();
#else
    __builtin_unreachable();
#endif
  }
  flow.PutNode(std::move(ptr));
}

void Converter::convertGatherAddAddNode(flow::Flow &flow,
                                        const graph::Graph &graph,
                                        const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::GatherAddAdd);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 4);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> gather_data = inputs[0];
  std::shared_ptr<graph::Edge> add0_weight = inputs[1];
  std::shared_ptr<graph::Edge> add1_weight = inputs[2];
  std::shared_ptr<graph::Edge> input = inputs[3];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(gather_data != nullptr);
  assert(add0_weight != nullptr);
  assert(add1_weight != nullptr);
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::GatherConstantDataTensorAddTensorLhsAddTensorLhsNode>
      ptr = nullptr;
  std::shared_ptr<graph::ConstantTensorEdge> gather_data_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(gather_data);
  std::shared_ptr<graph::ConstantTensorEdge> add0_weight_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(add0_weight);
  std::shared_ptr<graph::ConstantTensorEdge> add1_weight_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(add1_weight);
  std::shared_ptr<graph::NonConstantEdge> input_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(input);
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(gather_data_as_constant_tensor != nullptr);
  assert(add0_weight_as_constant_tensor != nullptr);
  assert(add1_weight_as_constant_tensor != nullptr);
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &input_name = input_as_non_constant->GetName();
  const std::string &output_name = output_as_non_constant->GetName();
  std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
  std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
  assert(input_ptr != nullptr);
  assert(output_ptr != nullptr);
#endif
  Tensor gather_data_tensor = gather_data_as_constant_tensor->GetValue();
  Tensor add0_weight_tensor = add0_weight_as_constant_tensor->GetValue();
  Tensor add1_weight_tensor = add1_weight_as_constant_tensor->GetValue();
  ptr = std::make_shared<
      flow::GatherConstantDataTensorAddTensorLhsAddTensorLhsNode>(
      std::move(name), std::move(gather_data_tensor),
      std::move(add0_weight_tensor), std::move(add1_weight_tensor),
      std::move(input_ptr), std::move(output_ptr));
  flow.PutNode(std::move(ptr));
}

void Converter::convertGemmNode(flow::Flow &flow, const graph::Graph &graph,
                                const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Gemm);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 3);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0];
  std::shared_ptr<graph::Edge> weights = inputs[1];
  std::shared_ptr<graph::Edge> bias = inputs[2];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(weights != nullptr);
  assert(bias != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::GemmNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> input_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(input);
  std::shared_ptr<graph::ConstantTensorEdge> weights_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(weights);
  std::shared_ptr<graph::ConstantTensorEdge> bias_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(bias);
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(weights_as_constant_tensor != nullptr);
  assert(bias_as_constant_tensor != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  float64_t alpha = 0;
  float64_t beta = 0;
  bool transA = false;
  bool transB = false;
  if (node.HasAttribute(flow::GemmNode::kAlphaAttrName)) {
    alpha = node.GetAttribute(flow::GemmNode::kAlphaAttrName).GetFloat32();
  }
  if (node.HasAttribute(flow::GemmNode::kBetaAttrName)) {
    beta = node.GetAttribute(flow::GemmNode::kBetaAttrName).GetFloat32();
  }
  if (node.HasAttribute(flow::GemmNode::kTransAAttrName)) {
    transA = node.GetAttribute(flow::GemmNode::kTransAAttrName).GetInt64();
  }
  if (node.HasAttribute(flow::GemmNode::kTransBAttrName)) {
    transB = node.GetAttribute(flow::GemmNode::kTransBAttrName).GetInt64();
  }
  const std::string &input_name = input_as_non_constant->GetName();
  const std::string &output_name = output_as_non_constant->GetName();
  std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
  std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
  Tensor weights_tensor = weights_as_constant_tensor->GetValue();
  Tensor biasTensor = bias_as_constant_tensor->GetValue();
  ptr = std::make_shared<flow::GemmConstantWeightsBiasNode>(
      std::move(name), std::move(input_ptr), std::move(weights_tensor),
      std::move(biasTensor), std::move(output_ptr), alpha, beta, transA,
      transB);
  flow.PutNode(std::move(ptr));
}

void Converter::convertLayerNormalizationNode(flow::Flow &flow,
                                              const graph::Graph &graph,
                                              const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::LayerNormalization);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 3);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0];
  std::shared_ptr<graph::Edge> scale = inputs[1];
  std::shared_ptr<graph::Edge> bias = inputs[2];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(scale != nullptr);
  assert(bias != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::LayerNormalizationNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> input_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(input);
  std::shared_ptr<graph::ConstantTensorEdge> scale_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(scale);
  std::shared_ptr<graph::ConstantTensorEdge> bias_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(bias);
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(scale_as_constant_tensor != nullptr);
  assert(bias_as_constant_tensor != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  int64_t axis = flow::LayerNormalizationNode::kAxis;
  float64_t epsilon = flow::LayerNormalizationNode::kEpsilon;
  if (node.HasAttribute(flow::LayerNormalizationNode::kAxisAttrName)) {
    axis = node.GetAttribute(flow::LayerNormalizationNode::kAxisAttrName)
               .GetInt64();
  }
  if (node.HasAttribute(flow::LayerNormalizationNode::kEpsilonAttrName)) {
    epsilon = node.GetAttribute(flow::LayerNormalizationNode::kEpsilonAttrName)
                  .GetFloat32();
  }
  const std::string &input_name = input_as_non_constant->GetName();
  const std::string &output_name = output_as_non_constant->GetName();
  std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
  std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
  assert(input_ptr != nullptr);
  assert(output_ptr != nullptr);
#endif
  Tensor scale_tensor = scale_as_constant_tensor->GetValue();
  Tensor biasTensor = bias_as_constant_tensor->GetValue();
  ptr = std::make_shared<flow::LayerNormalizationConstantScaleBiasNode>(
      std::move(name), std::move(input_ptr), std::move(scale_tensor),
      std::move(biasTensor), std::move(output_ptr), axis, epsilon);
  flow.PutNode(std::move(ptr));
}

void Converter::convertMatMulNode(flow::Flow &flow, const graph::Graph &graph,
                                  const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::MatMul);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> lhs = inputs[0];
  std::shared_ptr<graph::Edge> rhs = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(lhs != nullptr);
  assert(rhs != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(output_as_non_constant != nullptr);
#endif
  std::shared_ptr<flow::MatMulNode> ptr = nullptr;
  if (std::shared_ptr<graph::ConstantTensorEdge> lhs_as_constant_edge =
          std::dynamic_pointer_cast<graph::ConstantTensorEdge>(lhs)) {
    if (std::shared_ptr<graph::ConstantTensorEdge> rhs_as_constant_edge =
            std::dynamic_pointer_cast<graph::ConstantTensorEdge>(rhs)) {
// TODO: Currently, this code isn't expected to be reached, because such a case
// is expected to be optimized by the ONNX simplifier.
#ifdef DEBUG
      throw UnimplementedException();
#else
      __builtin_unreachable();
#endif
    } else if (std::shared_ptr<graph::NonConstantEdge> rhs_as_non_constantEdge =
                   std::dynamic_pointer_cast<graph::NonConstantEdge>(rhs)) {
      Tensor lhs_tensor = lhs_as_constant_edge->GetValue();
      const std::string &rhs_name = rhs_as_non_constantEdge->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> rhs_ptr = flow.GetEdge(rhs_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
      assert(rhs_ptr != nullptr);
      assert(output_ptr != nullptr);
#endif
      ptr = std::make_shared<flow::MatMulConstantLhsNode>(
          std::move(name), std::move(lhs_tensor), std::move(rhs_ptr),
          std::move(output_ptr));
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  } else if (std::shared_ptr<graph::NonConstantEdge> lhs_as_non_constantEdge =
                 std::dynamic_pointer_cast<graph::NonConstantEdge>(lhs)) {
    if (std::shared_ptr<graph::ConstantTensorEdge> rhs_as_constant_edge =
            std::dynamic_pointer_cast<graph::ConstantTensorEdge>(rhs)) {
      Tensor rhs_tensor = rhs_as_constant_edge->GetValue();
      const std::string &lhs_name = lhs_as_non_constantEdge->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> lhs_ptr = flow.GetEdge(lhs_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
      assert(lhs_ptr != nullptr);
      assert(output_ptr != nullptr);
#endif
      ptr = std::make_shared<flow::MatMulConstantRhsNode>(
          std::move(name), std::move(lhs_ptr), std::move(rhs_tensor),
          std::move(output_ptr));
    } else if (std::shared_ptr<graph::NonConstantEdge> rhs_as_non_constantEdge =
                   std::dynamic_pointer_cast<graph::NonConstantEdge>(rhs)) {
      const std::string &lhs_name = lhs_as_non_constantEdge->GetName();
      const std::string &rhs_name = rhs_as_non_constantEdge->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> lhs_ptr = flow.GetEdge(lhs_name);
      std::shared_ptr<flow::Edge> rhs_ptr = flow.GetEdge(rhs_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
      assert(lhs_ptr != nullptr);
      assert(rhs_ptr != nullptr);
      assert(output_ptr != nullptr);
#endif
      ptr = std::make_shared<flow::MatMulCommonNode>(
          std::move(name), std::move(lhs_ptr), std::move(rhs_ptr),
          std::move(output_ptr));
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  } else {
#ifdef DEBUG
    throw UnreachableException();
#else
    __builtin_unreachable();
#endif
  }
  flow.PutNode(std::move(ptr));
}

void Converter::convertMulNode(flow::Flow &flow, const graph::Graph &graph,
                               const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Mul);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input_lhs = inputs[0];
  std::shared_ptr<graph::Edge> input_rhs = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(input_lhs != nullptr);
  assert(input_rhs != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::MulNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  if (std::shared_ptr<graph::NonConstantEdge> input_lhs_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input_lhs)) {
    if (std::shared_ptr<graph::NonConstantEdge> input_rhs_as_non_constant =
            std::dynamic_pointer_cast<graph::NonConstantEdge>(input_rhs)) {
      const std::string &input_lhs_name = input_lhs_as_non_constant->GetName();
      const std::string &input_rhs_name = input_rhs_as_non_constant->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> input_lhs_tr = flow.GetEdge(input_lhs_name);
      std::shared_ptr<flow::Edge> input_rhs_ptr = flow.GetEdge(input_rhs_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
      assert(input_lhs_tr != nullptr);
      assert(input_rhs_ptr != nullptr);
      assert(output_ptr != nullptr);
#endif
      ptr = std::make_shared<flow::MulCommonNode>(
          std::move(name), std::move(input_lhs_tr), std::move(input_rhs_ptr),
          std::move(output_ptr));
    } else if (std::shared_ptr<graph::ConstantScalarEdge>
                   input_rhs_as_constant_scalar =
                       std::dynamic_pointer_cast<graph::ConstantScalarEdge>(
                           input_rhs)) {
      const std::string &input_name = input_lhs_as_non_constant->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
      assert(input_ptr != nullptr);
      assert(output_ptr != nullptr);
#endif
      ptr = std::make_shared<flow::MulConstantScalarNode>(
          std::move(name), std::move(input_ptr),
          input_rhs_as_constant_scalar->GetType(),
          input_rhs_as_constant_scalar->GetValue(), std::move(output_ptr));
    } else if (std::shared_ptr<graph::ConstantTensorEdge>
                   inputRhs_as_constant_tensor =
                       std::dynamic_pointer_cast<graph::ConstantTensorEdge>(
                           input_rhs)) {
      Tensor tensor = inputRhs_as_constant_tensor->GetValue();
      const std::string &input_name = input_lhs_as_non_constant->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
      assert(input_ptr != nullptr);
      assert(output_ptr != nullptr);
#endif
      ptr = std::make_shared<flow::MulConstantTensorNode>(
          std::move(name), std::move(input_ptr), std::move(tensor),
          std::move(output_ptr));
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  } else if (std::shared_ptr<graph::ConstantTensorEdge> input_lhs_as_constant =
                 std::dynamic_pointer_cast<graph::ConstantTensorEdge>(
                     input_lhs)) {
    if (std::shared_ptr<graph::NonConstantEdge> input_rhs_as_non_constant =
            std::dynamic_pointer_cast<graph::NonConstantEdge>(input_rhs)) {
      Tensor tensor = input_lhs_as_constant->GetValue();
      const std::string &input_name = input_rhs_as_non_constant->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
      assert(input_ptr != nullptr);
      assert(output_ptr != nullptr);
#endif
      ptr = std::make_shared<flow::MulConstantTensorNode>(
          std::move(name), std::move(input_ptr), std::move(tensor),
          std::move(output_ptr));
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  } else if (std::shared_ptr<graph::ConstantScalarEdge> input_lhs_as_constant =
                 std::dynamic_pointer_cast<graph::ConstantScalarEdge>(
                     input_lhs)) {
    if (std::shared_ptr<graph::NonConstantEdge> input_rhs_as_non_constant =
            std::dynamic_pointer_cast<graph::NonConstantEdge>(input_rhs)) {
      const std::string &input_name = input_rhs_as_non_constant->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
      ptr = std::make_shared<flow::MulConstantScalarNode>(
          std::move(name), std::move(input_ptr),
          input_lhs_as_constant->GetType(), input_lhs_as_constant->GetValue(),
          std::move(output_ptr));
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  } else {
#ifdef DEBUG
    throw UnreachableException();
#else
    __builtin_unreachable();
#endif
  }
  flow.PutNode(std::move(ptr));
}

void Converter::convertPowNode(flow::Flow &flow, const graph::Graph &graph,
                               const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Pow);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input_base = inputs[0];
  std::shared_ptr<graph::Edge> input_exponent = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(input_base != nullptr);
  assert(input_exponent != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::PowNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  if (std::shared_ptr<graph::NonConstantEdge> input_base_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input_base)) {
    if (std::shared_ptr<graph::ConstantScalarEdge>
            input_exponent_as_constant_scalar =
                std::dynamic_pointer_cast<graph::ConstantScalarEdge>(
                    input_exponent)) {
      const std::string &input_name = input_base_as_non_constant->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
      assert(input_ptr != nullptr);
      assert(output_ptr != nullptr);
#endif
      ptr = std::make_shared<flow::PowNode>(
          std::move(name), std::move(input_ptr),
          input_exponent_as_constant_scalar->GetType(),
          input_exponent_as_constant_scalar->GetValue(), std::move(output_ptr));
    }
  }
  flow.PutNode(std::move(ptr));
}

void Converter::convertReshapeNode(flow::Flow &flow, const graph::Graph &graph,
                                   const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Reshape);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0];
  std::shared_ptr<graph::Edge> shape = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(shape != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::ReshapeNode> ptr = nullptr;
  std::shared_ptr<graph::ConstantTensorEdge> shape_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(shape);
#ifdef DEBUG
  assert(isa<graph::NonConstantEdge>(input));
  assert(shape_as_constant_tensor != nullptr);
  assert(isa<graph::NonConstantEdge>(output));
#endif
  Tensor shape_tensor = shape_as_constant_tensor->GetValue();
  const std::string &input_name = input->GetName();
  const std::string &output_name = output->GetName();
  std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
  std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
  assert(input_ptr != nullptr);
  assert(output_ptr != nullptr);
  const Meta &output_meta = output_ptr->GetMeta();
  const std::vector<int64_t> &shape_vector = shape_tensor.GetShape();
  const std::vector<int64_t> &output_shape = output_meta.GetShape();
  assert(shape_vector.size() == 1);
  size_t size = shape_vector[0];
  assert(size == output_shape.size());
  std::vector<int64_t> shape_vec;
  for (size_t i = 0; i < size; ++i) {
    shape_vec.push_back(shape_tensor.Get({i}));
  }
  Meta expected_meta = Meta(output_meta.GetType(), std::move(shape_vec));
  std::optional<Meta> inferred_shape_opt = ReshapeShapeInference(
      expected_meta, std::accumulate(output_shape.begin(), output_shape.end(),
                                     1, std::multiplies<int64_t>()));
  assert(inferred_shape_opt.has_value());
  assert(output_meta == *inferred_shape_opt);
#endif
  ptr = std::make_shared<flow::ReshapeNode>(
      std::move(name), std::move(input_ptr), std::move(output_ptr));
  flow.PutNode(std::move(ptr));
}

void Converter::convertSoftmaxNode(flow::Flow &flow, const graph::Graph &graph,
                                   const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Softmax);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::SoftmaxNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> input_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(input);
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  int64_t axis = flow::SoftmaxNode::kAxis;
  if (node.HasAttribute(flow::SoftmaxNode::kAxisAttrName)) {
    axis = node.GetAttribute(flow::SoftmaxNode::kAxisAttrName).GetInt64();
  }
  const std::string &input_name = input_as_non_constant->GetName();
  const std::string &output_name = output_as_non_constant->GetName();
  std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
  std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
  assert(input_ptr != nullptr);
  assert(output_ptr != nullptr);
  const Meta &input_meta = input_ptr->GetMeta();
  const Meta &output_meta = output_ptr->GetMeta();
  assert(input_meta == output_meta);
#endif
  ptr = std::make_shared<flow::SoftmaxNode>(
      std::move(name), std::move(input_ptr), std::move(output_ptr), axis);
  flow.PutNode(std::move(ptr));
}

void Converter::convertSplitNode(flow::Flow &flow, const graph::Graph &graph,
                                 const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Split);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() >= 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0];
  std::shared_ptr<graph::Edge> shapes = inputs[1];
#ifdef DEBUG
  assert(input != nullptr);
#endif
  std::shared_ptr<flow::SplitNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> input_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(input);
  std::shared_ptr<graph::ConstantTensorEdge> shapes_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(shapes);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(shapes_as_constant_tensor != nullptr);
#endif
  const std::string &input_name = input_as_non_constant->GetName();
  std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
  Tensor shapes_tensor = shapes_as_constant_tensor->GetValue();
  const std::vector<int64_t> &shape = shapes_tensor.GetShape();
#ifdef DEBUG
  assert(shape.size() == 1);
#endif
  size_t size = shape[0];
#ifdef DEBUG
  assert(size == outputs.size());
#endif
  size_t axis = flow::SplitNode::kAxis;
  if (node.HasAttribute(flow::SplitNode::kAxisAttrName)) {
    axis = node.GetAttribute(flow::SplitNode::kAxisAttrName).GetInt64();
  }
  std::vector<std::shared_ptr<flow::Edge>> outputPtrs;
  for (size_t i = 0; i < size; ++i) {
    std::shared_ptr<graph::Edge> output = outputs[i];
    std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
        std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
    assert(output_as_non_constant != nullptr);
#endif
    const std::string &output_name = output_as_non_constant->GetName();
    std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
    assert(output_ptr != nullptr);
    const Meta &output_meta = output_ptr->GetMeta();
    std::vector<int64_t> output_shape = output_meta.GetShape();
    assert(output_shape[axis] == shapes_tensor.Get({i}));
#endif
    outputPtrs.push_back(std::move(output_ptr));
  }
  ptr = std::make_shared<flow::SplitNode>(std::move(name), std::move(input_ptr),
                                          std::move(outputPtrs), axis);
  flow.PutNode(std::move(ptr));
}

void Converter::convertSubNode(flow::Flow &flow, const graph::Graph &graph,
                               const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Sub);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input_lhs = inputs[0];
  std::shared_ptr<graph::Edge> input_rhs = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(input_lhs != nullptr);
  assert(input_rhs != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::SubNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  // The support format of Sub operator is limited to the following: the left
  // operator is a constant scalar, and the right operator is a non-constant
  // edge.
  // If new formats occur, the code should be updated.
  if (std::shared_ptr<graph::ConstantScalarEdge> input_lhs_as_constant_scalar =
          std::dynamic_pointer_cast<graph::ConstantScalarEdge>(input_lhs)) {
    if (std::shared_ptr<graph::NonConstantEdge> input_rhs_as_non_constant =
            std::dynamic_pointer_cast<graph::NonConstantEdge>(input_rhs)) {
      const std::string &input_name = input_rhs_as_non_constant->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
      std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
      assert(input_ptr != nullptr);
      assert(output_ptr != nullptr);
      const Meta &input_meta = input_ptr->GetMeta();
      const Meta &output_meta = output_ptr->GetMeta();
      assert(input_meta == output_meta);
#endif
      ptr = std::make_shared<flow::SubConstantScalarLhsNode>(
          std::move(name), std::move(input_ptr),
          input_lhs_as_constant_scalar->GetType(),
          input_lhs_as_constant_scalar->GetValue(), std::move(output_ptr));
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  } else {
#ifdef DEBUG
    throw UnreachableException();
#else
    __builtin_unreachable();
#endif
  }
  flow.PutNode(std::move(ptr));
}

void Converter::convertTanhNode(flow::Flow &flow, const graph::Graph &graph,
                                const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Tanh);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::TanhNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> input_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(input);
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &input_name = input_as_non_constant->GetName();
  const std::string &output_name = output_as_non_constant->GetName();
  std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
  std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
  assert(input_ptr != nullptr);
  assert(output_ptr != nullptr);
  const Meta &input_meta = input_ptr->GetMeta();
  const Meta &output_meta = output_ptr->GetMeta();
  assert(input_meta == output_meta);
#endif
  ptr = std::make_shared<flow::TanhNode>(std::move(name), std::move(input_ptr),
                                         std::move(output_ptr));
  flow.PutNode(std::move(ptr));
}

void Converter::convertTransposeNode(flow::Flow &flow,
                                     const graph::Graph &graph,
                                     const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Transpose);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::TransposeNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> input_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(input);
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &input_name = input_as_non_constant->GetName();
  const std::string &output_name = output_as_non_constant->GetName();
  std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
  std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
  assert(input_ptr != nullptr);
  assert(output_ptr != nullptr);
#endif
  const Meta &input_meta = input_ptr->GetMeta();
  const std::vector<int64_t> &input_shape = input_meta.GetShape();
  size_t inputShapeLen = input_shape.size();
  std::vector<int64_t> permutation(inputShapeLen);
  // the default permutation is the reverse of the input shape
  for (size_t i = 0; i < inputShapeLen; ++i) {
    permutation[i] = inputShapeLen - i - 1;
  }
  if (node.HasAttribute(flow::TransposeNode::kPermAttrName)) {
    const std::vector<int64_t> &perm =
        node.GetAttribute(flow::TransposeNode::kPermAttrName).GetInt64Array();
    permutation = perm;
  }
  ptr = std::make_shared<flow::TransposeNode>(
      std::move(name), std::move(input_ptr), std::move(output_ptr),
      std::move(permutation));
  flow.PutNode(std::move(ptr));
}

void Converter::convertUnsqueezeNode(flow::Flow &flow,
                                     const graph::Graph &graph,
                                     const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Unsqueeze);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0];
  std::shared_ptr<graph::Edge> axes = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(axes != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::UnsqueezeNode> ptr = nullptr;
  std::shared_ptr<graph::ConstantTensorEdge> axes_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(axes);
#ifdef DEBUG
  assert(axes_as_constant_tensor != nullptr);
  assert(isa<graph::NonConstantEdge>(input));
  assert(isa<graph::NonConstantEdge>(output));
#endif
  Tensor axesTensor = axes_as_constant_tensor->GetValue();
  const std::vector<int64_t> &axesVector = axesTensor.GetShape();
  const std::string &input_name = input->GetName();
  const std::string &output_name = output->GetName();
  std::shared_ptr<flow::Edge> input_ptr = flow.GetEdge(input_name);
  std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
  assert(input_ptr != nullptr);
  assert(output_ptr != nullptr);
  assert(axesVector.size() == 1);
#endif
  size_t size = axesVector[0];
  std::vector<int64_t> axesVec(size, 0);
  for (size_t i = 0; i < size; ++i) {
    axesVec[i] = axesTensor.Get({i});
  }
  ptr = std::make_shared<flow::UnsqueezeNode>(
      std::move(name), std::move(input_ptr), std::move(output_ptr),
      std::move(axesVec));
  flow.PutNode(std::move(ptr));
}

void Converter::convertWhereNode(flow::Flow &flow, const graph::Graph &graph,
                                 const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Where);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node);
  std::vector<std::shared_ptr<graph::Edge>> outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 3);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> condition = inputs[0];
  std::shared_ptr<graph::Edge> x = inputs[1];
  std::shared_ptr<graph::Edge> y = inputs[2];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(condition != nullptr);
  assert(x != nullptr);
  assert(y != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::WhereNode> ptr = nullptr;
  // The support format of Where operator is limited to the following: the
  // condition operator is a constant tensor, the x operator is a non-constant
  // edge, the y operator is a constant tensor, and the output operator is a
  // non-constant edge. If new formats occur, the code should be updated.
  std::shared_ptr<graph::ConstantTensorEdge> condition_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(condition);
#ifdef DEBUG
  assert(condition_as_constant_tensor != nullptr);
  assert(isa<graph::NonConstantEdge>(x));
  assert(isa<graph::NonConstantEdge>(output));
#endif
  if (std::shared_ptr<graph::ConstantScalarEdge> y_as_constant_scalar =
          std::dynamic_pointer_cast<graph::ConstantScalarEdge>(y)) {
    Tensor condition_tensor = condition_as_constant_tensor->GetValue();
    const std::string &x_name = x->GetName();
    const std::string &output_name = output->GetName();
    std::shared_ptr<flow::Edge> x_ptr = flow.GetEdge(x_name);
    std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
    assert(x_ptr != nullptr);
    assert(output_ptr != nullptr);
#endif
    ptr = std::make_shared<flow::WhereConstantCondConstantScalarYNode>(
        std::move(name), std::move(condition_tensor), std::move(x_ptr),
        y_as_constant_scalar->GetType(), y_as_constant_scalar->GetValue(),
        std::move(output_ptr));
  } else if (std::shared_ptr<graph::ConstantTensorEdge> y_as_constant_tensor =
                 std::dynamic_pointer_cast<graph::ConstantTensorEdge>(y)) {
    Tensor condition_tensor = condition_as_constant_tensor->GetValue();
    Tensor y_tensor = y_as_constant_tensor->GetValue();
    const std::string &x_name = x->GetName();
    const std::string &output_name = output->GetName();
    std::shared_ptr<flow::Edge> x_ptr = flow.GetEdge(x_name);
    std::shared_ptr<flow::Edge> output_ptr = flow.GetEdge(output_name);
#ifdef DEBUG
    assert(x_ptr != nullptr);
    assert(output_ptr != nullptr);
#endif
    ptr = std::make_shared<flow::WhereConstantCondConstantTensorYNode>(
        std::move(name), std::move(condition_tensor), std::move(x_ptr),
        std::move(y_tensor), std::move(output_ptr));
  } else {
#ifdef DEBUG
    throw UnreachableException();
#else
    __builtin_unreachable();
#endif
  }
  flow.PutNode(std::move(ptr));
}

} // namespace worker
} // namespace cpu_transformers
