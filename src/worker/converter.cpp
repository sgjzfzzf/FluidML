#include "worker/converter.h"
#include "structure/flow/edge.h"
#include "structure/flow/node.h"
#include "structure/flow/region.h"
#include "structure/graph/edge.h"
#include "structure/graph/node.h"
#include "utils/float.h"
#include "utils/isa.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#ifdef DEBUG
#include <cassert>
#endif

namespace fluidml {
namespace worker {

class ConverterImpl : public Converter {
public:
  ConverterImpl() = default;
  ConverterImpl(const ConverterImpl &converter) = delete;
  ConverterImpl(ConverterImpl &&converter) = default;
  virtual ~ConverterImpl() = default;
  flow::Flow Run(const graph::Graph &graph) override;

private:
  void convertAddNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertAddDivErfAddMulMulNode(flow::Flow &flow,
                                     const graph::Graph &graph,
                                     const graph::Node &node);
  void convertAveragePoolNode(flow::Flow &flow, const graph::Graph &graph,
                              const graph::Node &node);
  void convertCastNode(flow::Flow &flow, const graph::Graph &graph,
                       const graph::Node &node);
  void createClipNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertConcatNode(flow::Flow &flow, const graph::Graph &graph,
                         const graph::Node &node);
  void convertConvNode(flow::Flow &flow, const graph::Graph &graph,
                       const graph::Node &node);
  void convertCumSumNode(flow::Flow &flow, const graph::Graph &graph,
                         const graph::Node &node);
  void convertDivNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertDropoutNode(flow::Flow &flow, const graph::Graph &graph,
                          const graph::Node &node);
  void convertEqualNode(flow::Flow &flow, const graph::Graph &graph,
                        const graph::Node &node);
  void convertErfNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertFlattenNode(flow::Flow &flow, const graph::Graph &graph,
                          const graph::Node &node);
  void convertGatherNode(flow::Flow &flow, const graph::Graph &graph,
                         const graph::Node &node);
  void convertGatherAddAddNode(flow::Flow &flow, const graph::Graph &graph,
                               const graph::Node &node);
  void convertGemmNode(flow::Flow &flow, const graph::Graph &graph,
                       const graph::Node &node);
  void convertLayerNormalizationNode(flow::Flow &flow,
                                     const graph::Graph &graph,
                                     const graph::Node &node);
  void convertMatMulNode(flow::Flow &flow, const graph::Graph &graph,
                         const graph::Node &node);
  void convertMaxPoolNode(flow::Flow &flow, const graph::Graph &graph,
                          const graph::Node &node);
  void convertMulNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertNegNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertNotNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertPadNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertPowNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertReduceMeanNode(flow::Flow &flow, const graph::Graph &graph,
                             const graph::Node &node);
  void convertReluNode(flow::Flow &flow, const graph::Graph &graph,
                       const graph::Node &node);
  void convertReshapeNode(flow::Flow &flow, const graph::Graph &graph,
                          const graph::Node &node);
  void convertSliceNode(flow::Flow &flow, const graph::Graph &graph,
                        const graph::Node &node);
  void convertSoftmaxNode(flow::Flow &flow, const graph::Graph &graph,
                          const graph::Node &node);
  void convertSqrtNode(flow::Flow &flow, const graph::Graph &graph,
                       const graph::Node &node);
  void convertSqueezeNode(flow::Flow &flow, const graph::Graph &graph,
                          const graph::Node &node);
  void convertSubNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertTanhNode(flow::Flow &flow, const graph::Graph &graph,
                       const graph::Node &node);
  void convertTransposeNode(flow::Flow &flow, const graph::Graph &graph,
                            const graph::Node &node);
  void convertUnsqueezeNode(flow::Flow &flow, const graph::Graph &graph,
                            const graph::Node &node);
  void convertUnsqueezeSubMulNode(flow::Flow &flow, const graph::Graph &graph,
                                  const graph::Node &node);
  void convertWhereNode(flow::Flow &flow, const graph::Graph &graph,
                        const graph::Node &node);
};

std::unique_ptr<Converter> Converter::Make() {
  return std::make_unique<ConverterImpl>();
}

flow::Flow ConverterImpl::Run(const graph::Graph &graph) {
  flow::Flow flow;
  for (std::shared_ptr<graph::Edge> edge : graph.GetAllEdges()) {
    if (std::shared_ptr<graph::InputEdge> input_edge =
            std::dynamic_pointer_cast<graph::InputEdge>(edge)) {
      std::string name = input_edge->GetName();
      Meta meta = input_edge->GetMeta();
      std::shared_ptr<flow::InputRegion> region =
          std::make_shared<flow::InputRegion>(std::move(name), std::move(meta));
      flow.PutRegion(std::move(region));
    } else if (std::shared_ptr<graph::OutputEdge> output_edge =
                   std::dynamic_pointer_cast<graph::OutputEdge>(edge)) {
      std::string name = output_edge->GetName();
      std::shared_ptr<graph::Node> from = graph.GetEdgeFrom(*output_edge);
      std::string from_name = from->GetName();
      Meta meta = output_edge->GetMeta();
      std::shared_ptr<flow::OutputRegion> region =
          std::make_shared<flow::OutputRegion>(std::move(name),
                                               std::move(meta));
      std::shared_ptr<flow::OutputRegion> region_clone = region;
      region_clone = region;
      flow.PutRegion(std::move(region_clone));
    } else if (std::shared_ptr<graph::PureEdge> pure_edge =
                   std::dynamic_pointer_cast<graph::PureEdge>(edge)) {
      std::string name = pure_edge->GetName();
      Meta meta = pure_edge->GetMeta();
      std::shared_ptr<flow::InnerRegion> region =
          std::make_shared<flow::InnerRegion>(std::move(name), std::move(meta));
      std::shared_ptr<flow::InnerRegion> region_clone = region;
      flow.PutRegion(std::move(region_clone));
    } else if (std::shared_ptr<graph::ConstantTensorEdge> constant_edge =
                   std::dynamic_pointer_cast<graph::ConstantTensorEdge>(edge)) {
      std::string name = constant_edge->GetName();
      Tensor tensor = constant_edge->GetValue();
      std::shared_ptr<flow::ConstantRegion> region =
          std::make_shared<flow::ConstantRegion>(std::move(name),
                                                 std::move(tensor));
      flow.PutRegion(std::move(region));
    }
  }
  for (std::shared_ptr<graph::Node> node : graph.GetAllNodes()) {
    std::shared_ptr<flow::Node> ptr = nullptr;
    switch (node->GetOp()) {
    case graph::Node::Op::Add:
      convertAddNode(flow, graph, *node);
      break;
    case graph::Node::Op::AddDivErfAddMulMul:
      convertAddDivErfAddMulMulNode(flow, graph, *node);
      break;
    case graph::Node::Op::AveragePool:
      convertAveragePoolNode(flow, graph, *node);
      break;
    case graph::Node::Op::Cast:
      convertCastNode(flow, graph, *node);
      break;
    case graph::Node::Op::Clip:
      createClipNode(flow, graph, *node);
      break;
    case graph::Node::Op::Concat:
      convertConcatNode(flow, graph, *node);
      break;
    case graph::Node::Op::Conv:
      convertConvNode(flow, graph, *node);
      break;
    case graph::Node::Op::CumSum:
      convertCumSumNode(flow, graph, *node);
      break;
    case graph::Node::Op::Div:
      convertDivNode(flow, graph, *node);
      break;
    case graph::Node::Op::Dropout:
      convertDropoutNode(flow, graph, *node);
      break;
    case graph::Node::Op::Equal:
      convertEqualNode(flow, graph, *node);
      break;
    case graph::Node::Op::Erf:
      convertErfNode(flow, graph, *node);
      break;
    case graph::Node::Op::Flatten:
      convertFlattenNode(flow, graph, *node);
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
    case graph::Node::Op::MaxPool:
      convertMaxPoolNode(flow, graph, *node);
      break;
    case graph::Node::Op::Mul:
      convertMulNode(flow, graph, *node);
      break;
    case graph::Node::Op::Neg:
      convertNegNode(flow, graph, *node);
      break;
    case graph::Node::Op::Not:
      convertNotNode(flow, graph, *node);
      break;
    case graph::Node::Op::Pad:
      convertPadNode(flow, graph, *node);
      break;
    case graph::Node::Op::Pow:
      convertPowNode(flow, graph, *node);
      break;
    case graph::Node::Op::ReduceMean:
      convertReduceMeanNode(flow, graph, *node);
      break;
    case graph::Node::Op::Relu:
      convertReluNode(flow, graph, *node);
      break;
    case graph::Node::Op::Reshape:
      convertReshapeNode(flow, graph, *node);
      break;
    case graph::Node::Op::Slice:
      convertSliceNode(flow, graph, *node);
      break;
    case graph::Node::Op::Softmax:
      convertSoftmaxNode(flow, graph, *node);
      break;
    case graph::Node::Op::Sqrt:
      convertSqrtNode(flow, graph, *node);
      break;
    case graph::Node::Op::Squeeze:
      convertSqueezeNode(flow, graph, *node);
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
    case graph::Node::Op::UnsqueezeSubMul:
      convertUnsqueezeSubMulNode(flow, graph, *node);
      break;
    case graph::Node::Op::Where:
      convertWhereNode(flow, graph, *node);
      break;
    default:
#ifdef DEBUG
      assert(false && "unimplemented");
#else
      __builtin_unreachable();
#endif
    }
  }
  std::vector<std::shared_ptr<graph::Edge>> edges = graph.GetAllEdges();
  for (std::shared_ptr<graph::Edge> edge : edges) {
    if (std::shared_ptr<graph::InputEdge> input_edge =
            std::dynamic_pointer_cast<graph::InputEdge>(edge)) {
      std::string name = input_edge->GetName();
      std::vector<std::shared_ptr<graph::Node>> tos =
          graph.GetEdgeTo(*input_edge);
      for (std::shared_ptr<graph::Node> to : tos) {
        std::string to_name = to->GetName();
        std::shared_ptr<flow::Node> to_ptr = flow.GetNode(to_name);
        std::shared_ptr<flow::Region> region = flow.GetRegion(name);
        std::shared_ptr<flow::InputEdge> edge_ptr =
            std::make_shared<flow::InputEdge>(std::move(region),
                                              std::move(to_ptr));
        flow.PutEdge(std::move(edge_ptr));
      }
    } else if (std::shared_ptr<graph::OutputEdge> output_edge =
                   std::dynamic_pointer_cast<graph::OutputEdge>(edge)) {
      std::string name = output_edge->GetName();
      std::shared_ptr<graph::Node> from = graph.GetEdgeFrom(*output_edge);
      std::string from_name = from->GetName();
      std::shared_ptr<flow::Node> from_ptr = flow.GetNode(from_name);
      std::shared_ptr<flow::Region> region = flow.GetRegion(name);
      std::shared_ptr<flow::OutputEdge> edge_ptr =
          std::make_shared<flow::OutputEdge>(std::move(region),
                                             std::move(from_ptr));
      flow.PutEdge(std::move(edge_ptr));
      std::vector<std::shared_ptr<graph::Node>> tos =
          graph.GetEdgeTo(*output_edge);
      for (std::shared_ptr<graph::Node> to : tos) {
        from_name = from->GetName();
        std::string to_name = to->GetName();
        std::shared_ptr<flow::Node> from_ptr = flow.GetNode(from_name),
                                    to_ptr = flow.GetNode(to_name);
        region = flow.GetRegion(name);
        std::shared_ptr<flow::MemoryEdge> edge_ptr =
            std::make_shared<flow::MemoryEdge>(
                std::move(region), std::move(from_ptr), std::move(to_ptr));
        flow.PutEdge(std::move(edge_ptr));
      }
    } else if (std::shared_ptr<graph::PureEdge> pure_edge =
                   std::dynamic_pointer_cast<graph::PureEdge>(edge)) {
      std::string name = pure_edge->GetName();
      std::shared_ptr<graph::Node> from = graph.GetEdgeFrom(*pure_edge);
      std::vector<std::shared_ptr<graph::Node>> tos =
          graph.GetEdgeTo(*pure_edge);
      for (const std::shared_ptr<graph::Node> &to : tos) {
        std::string from_name = from->GetName(), to_name = to->GetName();
        std::shared_ptr<flow::Node> from_ptr = flow.GetNode(from_name),
                                    to_ptr = flow.GetNode(to_name);
        std::shared_ptr<flow::Region> region = flow.GetRegion(name);
        std::shared_ptr<flow::MemoryEdge> edge_ptr =
            std::make_shared<flow::MemoryEdge>(
                std::move(region), std::move(from_ptr), std::move(to_ptr));
        flow.PutEdge(std::move(edge_ptr));
      }
    } else if (std::shared_ptr<graph::ConstantTensorEdge> constant_edge =
                   std::dynamic_pointer_cast<graph::ConstantTensorEdge>(edge)) {
      std::string name = constant_edge->GetName();
      std::vector<std::shared_ptr<graph::Node>> tos =
          graph.GetEdgeTo(*constant_edge);
      for (const std::shared_ptr<graph::Node> &to : tos) {
        std::string to_name = to->GetName();
        std::shared_ptr<flow::Node> to_ptr = flow.GetNode(to_name);
        std::shared_ptr<flow::Region> region = flow.GetRegion(name);
        std::shared_ptr<flow::ConstantEdge> edge_ptr =
            std::make_shared<flow::ConstantEdge>(std::move(region),
                                                 std::move(to_ptr));
        flow.PutEdge(std::move(edge_ptr));
      }
    }
  }
#ifdef DEBUG
  assert(flow.Check());
#endif
  return flow;
}

void ConverterImpl::convertAddNode(flow::Flow &flow, const graph::Graph &graph,
                                   const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Add);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
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
  std::shared_ptr<flow::AddNode> ptr = nullptr;
  std::shared_ptr<graph::ConstantEdge> lhs_as_constant =
      std::dynamic_pointer_cast<graph::ConstantEdge>(lhs);
  std::shared_ptr<graph::ConstantEdge> rhs_as_constant =
      std::dynamic_pointer_cast<graph::ConstantEdge>(rhs);
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(output_as_non_constant != nullptr);
#endif
  Meta output_meta = output_as_non_constant->GetMeta();
  if (isa<graph::NonConstantEdge>(lhs) && isa<graph::NonConstantEdge>(rhs) ||
      isa<graph::ConstantTensorEdge>(lhs) && isa<graph::NonConstantEdge>(rhs) ||
      isa<graph::NonConstantEdge>(lhs) && isa<graph::ConstantTensorEdge>(rhs)) {
    const std::string &lhs_name = lhs->GetName();
    const std::string &rhs_name = rhs->GetName();
    const std::string &output_name = output->GetName();
    std::shared_ptr<flow::Region> lhs_region = flow.GetRegion(lhs_name);
    std::shared_ptr<flow::Region> rhs_region = flow.GetRegion(rhs_name);
    std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
    assert(lhs_region != nullptr);
    assert(rhs_region != nullptr);
    assert(output_region != nullptr);
    const Meta &lhs_meta = lhs_region->GetMeta();
    const Meta &rhs_meta = rhs_region->GetMeta();
    std::optional<Meta> output_meta_opt =
        BroadcastShape(lhs_meta, rhs_meta, output_meta.GetType());
    assert(output_meta_opt.has_value());
    assert(*output_meta_opt == output_meta);
#endif
    ptr = std::make_shared<flow::AddCommonNode>(
        std::move(name), std::move(lhs_region), std::move(rhs_region),
        std::move(output_region));
  } else if (isa<graph::NonConstantEdge>(lhs) &&
                 isa<graph::ConstantScalarEdge>(rhs) ||
             isa<graph::ConstantScalarEdge>(lhs) &&
                 isa<graph::NonConstantEdge>(rhs)) {
    std::shared_ptr<graph::NonConstantEdge> input_edge = nullptr;
    std::shared_ptr<graph::ConstantScalarEdge> weight_edge = nullptr;
    if (lhs_as_constant != nullptr && rhs_as_constant == nullptr) {
      input_edge = std::dynamic_pointer_cast<graph::NonConstantEdge>(rhs);
      weight_edge = std::dynamic_pointer_cast<graph::ConstantScalarEdge>(lhs);
    } else if (lhs_as_constant == nullptr && rhs_as_constant != nullptr) {
      input_edge = std::dynamic_pointer_cast<graph::NonConstantEdge>(lhs);
      weight_edge = std::dynamic_pointer_cast<graph::ConstantScalarEdge>(rhs);
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
#ifdef DEBUG
    assert(input_edge != nullptr);
    assert(weight_edge != nullptr);
#endif
    const std::string &input_name = input_edge->GetName();
    const std::string &output_name = output->GetName();
    std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
    std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
    assert(input_region != nullptr);
    assert(output_region != nullptr);
    Meta input_meta = input_region->GetMeta();
    assert(input_meta == output_meta);
#endif
    ptr = std::make_shared<flow::AddConstantNode>(
        std::move(name), weight_edge->GetType(), weight_edge->GetValue(),
        std::move(input_region), std::move(output_region));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertAddDivErfAddMulMulNode(flow::Flow &flow,
                                                  const graph::Graph &graph,
                                                  const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::AddDivErfAddMulMul);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 5);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> add0_edge = inputs[0];
  std::shared_ptr<graph::Edge> div_edge = inputs[1];
  std::shared_ptr<graph::Edge> add1_edge = inputs[2];
  std::shared_ptr<graph::Edge> mul1_edge = inputs[3];
  std::shared_ptr<graph::Edge> input_edge = inputs[4];
  std::shared_ptr<graph::Edge> output_edge = outputs[0];
#ifdef DEBUG
  assert(add0_edge != nullptr);
  assert(div_edge != nullptr);
  assert(add1_edge != nullptr);
  assert(mul1_edge != nullptr);
  assert(input_edge != nullptr);
  assert(output_edge != nullptr);
#endif
  std::shared_ptr<graph::ConstantTensorEdge> add0_weight =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(add0_edge);
  std::shared_ptr<graph::ConstantScalarEdge>
      div_weight =
          std::dynamic_pointer_cast<graph::ConstantScalarEdge>(div_edge),
      add1_weight =
          std::dynamic_pointer_cast<graph::ConstantScalarEdge>(add1_edge),
      mul1_weight =
          std::dynamic_pointer_cast<graph::ConstantScalarEdge>(mul1_edge);
  std::shared_ptr<graph::NonConstantEdge>
      input = std::dynamic_pointer_cast<graph::NonConstantEdge>(input_edge),
      output = std::dynamic_pointer_cast<graph::NonConstantEdge>(output_edge);
#ifdef DEBUG
  assert(add0_weight != nullptr);
  assert(div_weight != nullptr);
  assert(add1_weight != nullptr);
  assert(mul1_weight != nullptr);
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::AddDivErfAddMulMulNode> ptr = nullptr;
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
#endif
  Tensor add0_weight_tensor = add0_weight->GetValue();
  ptr = std::make_shared<flow::AddDivErfAddMulMulNode>(
      std::move(name), std::move(add0_weight_tensor), div_weight->GetType(),
      div_weight->GetValue(), add1_weight->GetType(), add1_weight->GetValue(),
      mul1_weight->GetType(), mul1_weight->GetValue(), std::move(input_region),
      std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertAveragePoolNode(flow::Flow &flow,
                                           const graph::Graph &graph,
                                           const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::AveragePool);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() >= 1);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::AveragePoolNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::vector<int64_t> &shape = input_as_non_constant->GetShape();
#ifdef DEBUG
  assert(shape.size() == output_as_non_constant->GetShape().size());
#endif
  const int64_t ndim = shape.size() - 2;
#ifdef DEBUG
  assert(ndim >= 1);
  assert(node.HasAttribute(flow::AveragePoolNode::kKernelShapeAttrName));
#endif
  std::vector<int64_t> dilations(ndim, 1),
      kernel_shape =
          node.GetAttribute(flow::AveragePoolNode::kKernelShapeAttrName)
              .GetInt64Array(),
      pads(2 * ndim, 0), strides(ndim, 1);
  if (node.HasAttribute(flow::AveragePoolNode::kDilationsAttrName)) {
    dilations = node.GetAttribute(flow::AveragePoolNode::kDilationsAttrName)
                    .GetInt64Array();
  }
  if (node.HasAttribute(flow::AveragePoolNode::kPadsAttrName)) {
    pads =
        node.GetAttribute(flow::AveragePoolNode::kPadsAttrName).GetInt64Array();
  }
  if (node.HasAttribute(flow::AveragePoolNode::kStridesAttrName)) {
    strides = node.GetAttribute(flow::AveragePoolNode::kStridesAttrName)
                  .GetInt64Array();
  }
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
#endif
  if (std::all_of(pads.begin(), pads.end(),
                  [](int64_t pad) { return pad == 0; })) {
    ptr = std::make_shared<flow::AveragePoolWithoutPaddingNode>(
        std::move(name), std::move(dilations), std::move(kernel_shape),
        std::move(strides), std::move(input_region), std::move(output_region));
  } else {
#ifdef DEBUG
    assert(false && "unimplemented");
#else
    __builtin_unreachable();
#endif
  }
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertCastNode(flow::Flow &flow, const graph::Graph &graph,
                                    const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Cast);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::CastNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
  const Meta &input_meta = input_region->GetMeta(),
             &output_meta = output_region->GetMeta();
  const std::vector<int64_t> &input_shape = input_meta.GetShape(),
                             &output_shape = output_meta.GetShape();
  assert(input_shape == output_shape);
#endif
  ptr = std::make_shared<flow::CastNode>(
      std::move(name), std::move(input_region), std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::createClipNode(flow::Flow &flow, const graph::Graph &graph,
                                   const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Clip);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() >= 1);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::ClipNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  float32_t min = 0, max = 0;
  if (inputs.size() == 3) {
    std::shared_ptr<graph::Edge> min_edge = inputs[1], max_edge = inputs[2];
#ifdef DEBUG
    assert(min_edge != nullptr);
    assert(max_edge != nullptr);
#endif
    std::shared_ptr<graph::ConstantScalarEdge>
        min_as_constant =
            std::dynamic_pointer_cast<graph::ConstantScalarEdge>(min_edge),
        max_as_constant =
            std::dynamic_pointer_cast<graph::ConstantScalarEdge>(max_edge);
#ifdef DEBUG
    assert(min_as_constant != nullptr);
    assert(max_as_constant != nullptr);
#endif
    min = min_as_constant->GetValue();
    max = max_as_constant->GetValue();
  } else {
#ifdef DEBUG
    assert(inputs.size() == 1);
    assert(node.HasAttribute(flow::ClipNode::kMinAttrName));
    assert(node.HasAttribute(flow::ClipNode::kMaxAttrName));
#endif
    min = node.GetAttribute(flow::ClipNode::kMinAttrName).GetFloat32();
    max = node.GetAttribute(flow::ClipNode::kMaxAttrName).GetFloat32();
  }
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
#endif
  const Meta &input_meta = input_region->GetMeta(),
             &output_meta = output_region->GetMeta();
  const std::vector<int64_t> &input_shape = input_meta.GetShape(),
                             &output_shape = output_meta.GetShape();
#ifdef DEBUG
  assert(input_shape == output_shape);
#endif
  ptr = std::make_shared<flow::ClipNode>(std::move(name), min, max,
                                         std::move(input_region),
                                         std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertConcatNode(flow::Flow &flow,
                                      const graph::Graph &graph,
                                      const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Concat);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> lhs = inputs[0], rhs = inputs[1],
                               output = outputs[0];
#ifdef DEBUG
  assert(lhs != nullptr);
  assert(rhs != nullptr);
  assert(output != nullptr);
#endif
  const size_t axis =
      node.GetAttribute(flow::ConcatNode::kAxisAttrName).GetInt64();
  std::shared_ptr<flow::ConcatNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      lhs_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(lhs),
      rhs_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(rhs),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(lhs_as_non_constant != nullptr);
  assert(rhs_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &lhs_name = lhs->GetName(), &rhs_name = rhs->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> lhs_region = flow.GetRegion(lhs_name),
                                rhs_region = flow.GetRegion(rhs_name),
                                output_region = flow.GetRegion(output_name);
  ptr = std::make_shared<flow::Concat2CommonNode>(
      std::move(name), axis, std::move(lhs_region), std::move(rhs_region),
      std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertConvNode(flow::Flow &flow, const graph::Graph &graph,
                                    const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Conv);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(outputs.size() == 1);
#endif
  std::optional<Tensor> bias = std::nullopt;
  if (inputs.size() == 3) {
    std::shared_ptr<graph::Edge> bias_edge = inputs[2];
#ifdef DEBUG
    assert(bias_edge != nullptr);
#endif
    std::shared_ptr<graph::ConstantTensorEdge> bias_as_constant =
        std::dynamic_pointer_cast<graph::ConstantTensorEdge>(bias_edge);
#ifdef DEBUG
    assert(bias_as_constant != nullptr);
#endif
    bias = bias_as_constant->GetValue();
  } else {
#ifdef DEBUG
    assert(inputs.size() == 2);
#endif
  }
  std::shared_ptr<graph::Edge> input = inputs[0], weight = inputs[1],
                               output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(weight != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::ConvNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  std::shared_ptr<graph::ConstantTensorEdge> weight_as_constant =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(weight);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(weight_as_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &input_name = input->GetName(),
                    &weight_name = weight->GetName(),
                    &output_name = output->GetName();
  const Meta &input_meta = input_as_non_constant->GetMeta(),
             &weight_meta = weight_as_constant->GetMeta(),
             &output_meta = output_as_non_constant->GetMeta();
  const std::vector<int64_t> &input_shape = input_meta.GetShape(),
                             &weight_shape = weight_meta.GetShape(),
                             &output_shape = output_meta.GetShape();
  const size_t rank = input_shape.size();
#ifdef DEBUG
  assert(rank >= 3);
  assert(weight_shape.size() == rank);
  assert(output_shape.size() == rank);
#endif
  size_t group = 1;
  if (node.HasAttribute(flow::ConvNode::kGroupAttrName)) {
    group = node.GetAttribute(flow::ConvNode::kGroupAttrName).GetInt64();
  }
#ifdef DEBUG
  assert(node.HasAttribute(flow::ConvNode::kDilationsAttrName));
  assert(node.HasAttribute(flow::ConvNode::kKernelShapeAttrName));
  assert(node.HasAttribute(flow::ConvNode::kPadsAttrName));
  assert(node.HasAttribute(flow::ConvNode::kStridesAttrName));
#endif
  std::vector<int64_t>
      dilations =
          node.GetAttribute(flow::ConvNode::kDilationsAttrName).GetInt64Array(),
      kernel_shape = node.GetAttribute(flow::ConvNode::kKernelShapeAttrName)
                         .GetInt64Array(),
      pads = node.GetAttribute(flow::ConvNode::kPadsAttrName).GetInt64Array(),
      strides =
          node.GetAttribute(flow::ConvNode::kStridesAttrName).GetInt64Array();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                weight_region = flow.GetRegion(weight_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(weight_region != nullptr);
  assert(output_region != nullptr);
#endif
  if (std::all_of(pads.begin(), pads.end(),
                  [](int64_t pad) { return pad == 0; })) {
    ptr = std::make_shared<flow::ConvWithoutPaddingNode>(
        std::move(name), std::move(dilations), group, std::move(kernel_shape),
        std::move(strides), std::move(bias), std::move(input_region),
        std::move(weight_region), std::move(output_region));
  } else {
    ptr = std::make_shared<flow::ConvWithPaddingNode>(
        std::move(name), std::move(dilations), group, std::move(kernel_shape),
        std::move(pads), std::move(strides), std::move(bias),
        std::move(input_region), std::move(weight_region),
        std::move(output_region));
  }
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertCumSumNode(flow::Flow &flow,
                                      const graph::Graph &graph,
                                      const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::CumSum);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], axis = inputs[1],
                               output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(axis != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::CumSumNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  std::shared_ptr<graph::ConstantScalarEdge> axis_as_constant_scalar =
      std::dynamic_pointer_cast<graph::ConstantScalarEdge>(axis);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(axis_as_constant_scalar != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  int64_t axis_value = axis_as_constant_scalar->GetValue();
  bool exclusive =
      node.HasAttribute(flow::CumSumNode::kExclusiveAttrName)
          ? node.GetAttribute(flow::CumSumNode::kExclusiveAttrName).GetInt64()
          : flow::CumSumNode::kExclusive;
  bool reverse =
      node.HasAttribute(flow::CumSumNode::kReverseAttrName)
          ? node.GetAttribute(flow::CumSumNode::kReverseAttrName).GetInt64()
          : flow::CumSumNode::kReverse;
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
  const Meta &input_meta = input_region->GetMeta(),
             &output_meta = output_region->GetMeta();
  assert(input_meta == output_meta);
#endif
  ptr = std::make_shared<flow::CumSumNode>(
      std::move(name), axis_value, exclusive, reverse, std::move(input_region),
      std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertDivNode(flow::Flow &flow, const graph::Graph &graph,
                                   const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Div);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> lhs = inputs[0], rhs = inputs[1],
                               output = outputs[0];
#ifdef DEBUG
  assert(lhs != nullptr);
  assert(rhs != nullptr);
#endif
  std::shared_ptr<flow::DivNode> ptr = nullptr;
  // The support format of Div operator is limited to the following:
  // the left operator is a tensor, and it's a non-constant edge
  // the right operator is a scalar, and it's a constant scalar edge
  // If new formats occur, the code should be updated.
  if (std::shared_ptr<graph::NonConstantEdge> lhs_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(lhs)) {
    if (std::shared_ptr<graph::ConstantScalarEdge> rhs_as_constant_scalar =
            std::dynamic_pointer_cast<graph::ConstantScalarEdge>(rhs)) {
      const std::string &input_name = lhs->GetName(),
                        &output_name = output->GetName();
      std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                    output_region = flow.GetRegion(output_name);
#ifdef DEBUG
      assert(input_region != nullptr);
      assert(output_region != nullptr);
      const Meta &input_meta = input_region->GetMeta(),
                 &output_meta = output_region->GetMeta();
      assert(input_meta == output_meta);
#endif
      ptr = std::make_shared<flow::DivConstantRhsNode>(
          std::move(name), rhs_as_constant_scalar->GetType(),
          rhs_as_constant_scalar->GetValue(), std::move(input_region),
          std::move(output_region));
    } else if (std::shared_ptr<graph::NonConstantEdge> rhs_as_non_constant =
                   std::dynamic_pointer_cast<graph::NonConstantEdge>(rhs)) {
      const std::string &lhs_name = lhs->GetName(), &rhs_name = rhs->GetName(),
                        &output_name = output->GetName();
      std::shared_ptr<flow::Region> lhs_region = flow.GetRegion(lhs_name),
                                    rhs_region = flow.GetRegion(rhs_name),
                                    output_region = flow.GetRegion(output_name);
#ifdef DEBUG
      assert(lhs_region != nullptr);
      assert(rhs_region != nullptr);
      assert(output_region != nullptr);
#endif
      ptr = std::make_shared<flow::DivCommonNode>(
          std::move(name), std::move(lhs_region), std::move(rhs_region),
          std::move(output_region));
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
  } else {
#ifdef DEBUG
    assert(false && "unimplemented");
#else
    __builtin_unreachable();
#endif
  }
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertDropoutNode(flow::Flow &flow,
                                       const graph::Graph &graph,
                                       const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Dropout);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
  float64_t ratio = flow::DropoutNode::kRatio;
  if (inputs.size() == 2) {
    std::shared_ptr<graph::Edge> ratio_edge = inputs[1];
#ifdef DEBUG
    assert(ratio_edge != nullptr);
#endif
    std::shared_ptr<graph::ConstantScalarEdge> ratio_as_constant_scalar =
        std::dynamic_pointer_cast<graph::ConstantScalarEdge>(ratio_edge);
#ifdef DEBUG
    assert(ratio_as_constant_scalar != nullptr);
#endif
    ratio = ratio_as_constant_scalar->GetValue();
  } else if (inputs.size() == 1) {
    if (node.HasAttribute(flow::DropoutNode::kRatioAttrName)) {
      ratio = node.GetAttribute(flow::DropoutNode::kRatioAttrName).GetFloat32();
    }
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  std::shared_ptr<graph::Edge> input = inputs[0], output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::DropoutNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
  const Meta &input_meta = input_region->GetMeta(),
             &output_meta = output_region->GetMeta();
  assert(input_meta == output_meta);
#endif
  ptr = std::make_shared<flow::DropoutNode>(std::move(name), ratio,
                                            std::move(input_region),
                                            std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertEqualNode(flow::Flow &flow,
                                     const graph::Graph &graph,
                                     const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Equal);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], val = inputs[1],
                               output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::EqualNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  std::shared_ptr<graph::ConstantScalarEdge> val_as_constant_scalar =
      std::dynamic_pointer_cast<graph::ConstantScalarEdge>(val);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(val_as_constant_scalar != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  Type type = val_as_constant_scalar->GetType();
  float64_t value = val_as_constant_scalar->GetValue();
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
#endif
  ptr = std::make_shared<flow::EqualNode>(std::move(name), type, value,
                                          std::move(input_region),
                                          std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertErfNode(flow::Flow &flow, const graph::Graph &graph,
                                   const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Erf);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
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
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
  std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
  Meta input_meta = input_region->GetMeta();
  const Meta &output_meta = output_region->GetMeta();
  assert(input_meta == output_meta);
#endif
  ptr = std::make_shared<flow::ErfNode>(
      std::move(name), std::move(input_region), std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertFlattenNode(flow::Flow &flow,
                                       const graph::Graph &graph,
                                       const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Flatten);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::FlattenNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
#endif
  int64_t axis = flow::FlattenNode::kAxis;
  if (node.HasAttribute(flow::FlattenNode::kAxisAttrName)) {
    axis = node.GetAttribute(flow::FlattenNode::kAxisAttrName).GetInt64();
  }
  ptr = std::make_shared<flow::FlattenNode>(
      std::move(name), axis, std::move(input_region), std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertGatherNode(flow::Flow &flow,
                                      const graph::Graph &graph,
                                      const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Gather);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input_lhs = inputs[0], input_rhs = inputs[1],
                               output = outputs[0];
#ifdef DEBUG
  assert(input_lhs != nullptr);
  assert(input_rhs != nullptr);
  assert(output != nullptr);
#endif
  size_t axis = 0;
  if (node.HasAttribute(flow::GatherNode::kAxisAttrName)) {
    axis = node.GetAttribute(flow::GatherNode::kAxisAttrName).GetInt64();
  }
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
      std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
      std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
      assert(input_region != nullptr);
      assert(output_region != nullptr);
#endif
      ptr = std::make_shared<flow::GatherConstantIndexScalarNode>(
          std::move(name), std::move(input_region), std::move(output_region),
          std::lround(input_rhs_as_constant_scalar->GetValue()), axis);
    } else if (std::shared_ptr<graph::ConstantTensorEdge>
                   input_rhs_as_constant_tensor =
                       std::dynamic_pointer_cast<graph::ConstantTensorEdge>(
                           input_rhs)) {
      const std::string &input_name = input_lhs_as_non_constant->GetName();
      const std::string &output_name = output->GetName();
      std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
      std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
      assert(input_region != nullptr);
      assert(output_region != nullptr);
#endif
      Tensor tensor = input_rhs_as_constant_tensor->GetValue();
      ptr = std::make_shared<flow::GatherConstantIndicesTensorNode>(
          std::move(name), std::move(input_region), std::move(output_region),
          std::move(tensor), axis);
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
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
      std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
      std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
      assert(input_region != nullptr);
      assert(output_region != nullptr);
#endif
      ptr = std::make_shared<flow::GatherConstantDataTensorNode>(
          std::move(name), std::move(input_region), std::move(output_region),
          std::move(tensor), axis);
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertGatherAddAddNode(flow::Flow &flow,
                                            const graph::Graph &graph,
                                            const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::GatherAddAdd);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
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
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
  std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
#endif
  Tensor gather_data_tensor = gather_data_as_constant_tensor->GetValue();
  Tensor add0_weight_tensor = add0_weight_as_constant_tensor->GetValue();
  Tensor add1_weight_tensor = add1_weight_as_constant_tensor->GetValue();
  ptr = std::make_shared<
      flow::GatherConstantDataTensorAddTensorLhsAddTensorLhsNode>(
      std::move(name), std::move(gather_data_tensor),
      std::move(add0_weight_tensor), std::move(add1_weight_tensor),
      std::move(input_region), std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertGemmNode(flow::Flow &flow, const graph::Graph &graph,
                                    const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Gemm);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 3);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> lhs = inputs[0];
  std::shared_ptr<graph::Edge> rhs = inputs[1];
  std::shared_ptr<graph::Edge> bias = inputs[2];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(lhs != nullptr);
  assert(rhs != nullptr);
  assert(bias != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::GemmNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> lhs_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(lhs);
  std::shared_ptr<graph::ConstantTensorEdge> rhs_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(rhs);
  std::shared_ptr<graph::ConstantTensorEdge> bias_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(bias);
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(lhs_as_non_constant != nullptr);
  assert(rhs_as_constant_tensor != nullptr);
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
  const std::string &lhs_name = lhs_as_non_constant->GetName(),
                    &rhs_name = rhs_as_constant_tensor->GetName(),
                    &output_name = output_as_non_constant->GetName();
  std::shared_ptr<flow::Region> lhs_region = flow.GetRegion(lhs_name),
                                rhs_region = flow.GetRegion(rhs_name),
                                output_region = flow.GetRegion(output_name);
  Tensor biasTensor = bias_as_constant_tensor->GetValue();
  ptr = std::make_shared<flow::GemmConstantWeightsBiasNode>(
      std::move(name), std::move(lhs_region), std::move(rhs_region),
      std::move(output_region), std::move(biasTensor), alpha, beta, transA,
      transB);
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertLayerNormalizationNode(flow::Flow &flow,
                                                  const graph::Graph &graph,
                                                  const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::LayerNormalization);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
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
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
  std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
#endif
  Tensor scale_tensor = scale_as_constant_tensor->GetValue();
  Tensor biasTensor = bias_as_constant_tensor->GetValue();
  ptr = std::make_shared<flow::LayerNormalizationConstantScaleBiasNode>(
      std::move(name), std::move(scale_tensor), std::move(biasTensor),
      std::move(input_region), std::move(output_region), axis, epsilon);
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertMatMulNode(flow::Flow &flow,
                                      const graph::Graph &graph,
                                      const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::MatMul);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
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
  assert(isa<graph::NonConstantEdge>(output));
  assert(isa<graph::NonConstantEdge>(lhs) || isa<graph::NonConstantEdge>(rhs));
#endif
  const std::string &lhs_name = lhs->GetName(), &rhs_name = rhs->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> lhs_region = flow.GetRegion(lhs_name),
                                rhs_region = flow.GetRegion(rhs_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(lhs_region != nullptr);
  assert(rhs_region != nullptr);
  assert(output_region != nullptr);
#endif
  std::shared_ptr<flow::MatMulNode> ptr = std::make_shared<flow::MatMulNode>(
      std::move(name), std::move(lhs_region), std::move(rhs_region),
      std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertMaxPoolNode(flow::Flow &flow,
                                       const graph::Graph &graph,
                                       const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::MaxPool);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::MaxPoolNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
  assert(node.HasAttribute(flow::MaxPoolNode::kKernelShapeAttrName));
  assert(node.HasAttribute(flow::MaxPoolNode::kPadsAttrName));
  assert(node.HasAttribute(flow::MaxPoolNode::kStridesAttrName));
#endif
  std::vector<int64_t>
      kernel_shape = node.GetAttribute(flow::MaxPoolNode::kKernelShapeAttrName)
                         .GetInt64Array(),
      pads =
          node.GetAttribute(flow::MaxPoolNode::kPadsAttrName).GetInt64Array(),
      strides = node.GetAttribute(flow::MaxPoolNode::kStridesAttrName)
                    .GetInt64Array();
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
  if (std::all_of(pads.begin(), pads.end(),
                  [](int64_t pad) { return pad == 0; })) {
    ptr = std::make_shared<flow::MaxPoolWithoutPaddingNode>(
        std::move(name), std::move(kernel_shape), std::move(strides),
        std::move(input_region), std::move(output_region));
  } else {
#ifdef DEBUG
    assert(false && "unimplemented");
#else
    __builtin_unreachable();
#endif
  }
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertMulNode(flow::Flow &flow, const graph::Graph &graph,
                                   const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Mul);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
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
  std::shared_ptr<flow::MulNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  if (isa<graph::NonConstantEdge>(lhs) && isa<graph::NonConstantEdge>(rhs) ||
      isa<graph::NonConstantEdge>(lhs) && isa<graph::ConstantTensorEdge>(rhs) ||
      isa<graph::ConstantTensorEdge>(lhs) && isa<graph::NonConstantEdge>(rhs)) {
    const std::string &lhs_name = lhs->GetName(), &rhs_name = rhs->GetName(),
                      &output_name = output->GetName();
    std::shared_ptr<flow::Region> lhs_region = flow.GetRegion(lhs_name),
                                  rhs_region = flow.GetRegion(rhs_name),
                                  output_region = flow.GetRegion(output_name);
#ifdef DEBUG
    assert(lhs_region != nullptr);
    assert(rhs_region != nullptr);
    assert(output_region != nullptr);
#endif
    ptr = std::make_shared<flow::MulCommonNode>(
        std::move(name), std::move(lhs_region), std::move(rhs_region),
        std::move(output_region));
  } else if (isa<graph::NonConstantEdge>(lhs) &&
                 isa<graph::ConstantScalarEdge>(rhs) ||
             isa<graph::ConstantScalarEdge>(lhs) &&
                 isa<graph::NonConstantEdge>(rhs)) {
    std::shared_ptr<graph::NonConstantEdge> input_edge = nullptr;
    std::shared_ptr<graph::ConstantScalarEdge> weight_edge = nullptr;
    if (std::dynamic_pointer_cast<graph::ConstantScalarEdge>(lhs) != nullptr &&
        std::dynamic_pointer_cast<graph::NonConstantEdge>(rhs) != nullptr) {
      input_edge = std::dynamic_pointer_cast<graph::NonConstantEdge>(rhs);
      weight_edge = std::dynamic_pointer_cast<graph::ConstantScalarEdge>(lhs);
    } else if (std::dynamic_pointer_cast<graph::NonConstantEdge>(lhs) !=
                   nullptr &&
               std::dynamic_pointer_cast<graph::ConstantScalarEdge>(rhs) !=
                   nullptr) {
      input_edge = std::dynamic_pointer_cast<graph::NonConstantEdge>(lhs);
      weight_edge = std::dynamic_pointer_cast<graph::ConstantScalarEdge>(rhs);
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
#ifdef DEBUG
    assert(input_edge != nullptr);
    assert(weight_edge != nullptr);
#endif
    const std::string &input_name = input_edge->GetName(),
                      &output_name = output->GetName();
    std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                  output_region = flow.GetRegion(output_name);
#ifdef DEBUG
    assert(input_region != nullptr);
    assert(output_region != nullptr);
#endif
    ptr = std::make_shared<flow::MulConstantNode>(
        std::move(name), std::move(input_region), weight_edge->GetType(),
        weight_edge->GetValue(), std::move(output_region));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertNegNode(flow::Flow &flow, const graph::Graph &graph,
                                   const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Neg);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
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
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
#endif
  std::shared_ptr<flow::NegNode> ptr = std::make_shared<flow::NegNode>(
      std::move(name), std::move(input_region), std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertNotNode(flow::Flow &flow, const graph::Graph &graph,
                                   const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Not);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
#endif
  std::shared_ptr<flow::NotNode> ptr = std::make_shared<flow::NotNode>(
      std::move(name), std::move(input_region), std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertPadNode(flow::Flow &flow, const graph::Graph &graph,
                                   const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Pad);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], pads = inputs[1],
                               output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(pads != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::PadNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  std::shared_ptr<graph::ConstantTensorEdge> pads_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(pads);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(pads_as_constant_tensor != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  std::vector<std::tuple<int64_t, int64_t>> pads_vector;
  Tensor pads_tensor = pads_as_constant_tensor->GetValue();
  const std::vector<int64_t> &pads_shape = pads_tensor.GetShape();
#ifdef DEBUG
  assert(pads_tensor.GetType() == Type::kInt64);
  assert(pads_shape.size() == 1);
#endif
  const size_t size = pads_shape[0];
#ifdef DEBUG
  assert(size % 2 == 0);
#endif
  for (size_t i = 0; i < size / 2; ++i) {
    pads_vector.push_back(
        {static_cast<int64_t>(pads_tensor.Get({i})),
         static_cast<int64_t>(pads_tensor.Get({i + size / 2}))});
  }
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
#endif
  ptr = std::make_shared<flow::PadNode>(std::move(name), std::move(pads_vector),
                                        std::move(input_region),
                                        std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertPowNode(flow::Flow &flow, const graph::Graph &graph,
                                   const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Pow);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input_base = inputs[0],
                               input_exponent = inputs[1], output = outputs[0];
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
      std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
      std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
      assert(input_region != nullptr);
      assert(output_region != nullptr);
#endif
      ptr = std::make_shared<flow::PowNode>(
          std::move(name), input_exponent_as_constant_scalar->GetType(),
          input_exponent_as_constant_scalar->GetValue(),
          std::move(input_region), std::move(output_region));
    }
  }
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertReduceMeanNode(flow::Flow &flow,
                                          const graph::Graph &graph,
                                          const graph::Node &node) {
  // There is a difference after the update of ONNX version 18, so check it
  // dynamically.
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::ReduceMean);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
  std::vector<int64_t> axes;
  if (inputs.size() == 2) {
    // If the onnx version is equal or larger than 18, "axes" is an input.
    std::shared_ptr<graph::Edge> axes_edge = inputs[1];
    std::shared_ptr<graph::ConstantTensorEdge> axes_as_constant_tensor =
        std::dynamic_pointer_cast<graph::ConstantTensorEdge>(axes_edge);
#ifdef DEBUG
    assert(axes_as_constant_tensor != nullptr);
#endif
    Tensor axes_tensor = axes_as_constant_tensor->GetValue();
    const std::vector<int64_t> &axes_shape = axes_tensor.GetShape();
#ifdef DEBUG
    assert(axes_tensor.GetType() == Type::kInt64);
    assert(axes_shape.size() == 1);
#endif
    const size_t size = axes_shape[0];
    for (size_t i = 0; i < size; ++i) {
      axes.push_back(static_cast<int64_t>(axes_tensor.Get({i})));
    }
  } else if (inputs.size() == 1) {
    // If the onnx version is less than 18, "axes" is an attribute.
    axes = node.GetAttribute("axes").GetInt64Array();
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
#ifdef DEBUG
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  bool keepdims = node.GetAttribute("keepdims").GetInt64();
  std::shared_ptr<flow::ReduceMeanNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
  ptr = std::make_shared<flow::ReduceMeanNode>(
      std::move(name), std::move(input_region), std::move(output_region),
      std::move(axes), keepdims);
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertReluNode(flow::Flow &flow, const graph::Graph &graph,
                                    const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Relu);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::ReluNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
#endif
  ptr = std::make_shared<flow::ReluNode>(
      std::move(name), std::move(input_region), std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertReshapeNode(flow::Flow &flow,
                                       const graph::Graph &graph,
                                       const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Reshape);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
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
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
  std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
  const Meta &output_meta = output_region->GetMeta();
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
      std::move(name), std::move(input_region), std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertSliceNode(flow::Flow &flow,
                                     const graph::Graph &graph,
                                     const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Slice);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 5);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], begins = inputs[1],
                               ends = inputs[2], axes = inputs[3],
                               steps = inputs[4], output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(begins != nullptr);
  assert(ends != nullptr);
  assert(axes != nullptr);
  assert(steps != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::SliceNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  std::shared_ptr<graph::ConstantTensorEdge> begins_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(begins);
  std::shared_ptr<graph::ConstantTensorEdge> ends_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(ends);
  std::shared_ptr<graph::ConstantTensorEdge> axes_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(axes);
  std::shared_ptr<graph::ConstantTensorEdge> steps_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(steps);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
  assert(begins_as_constant_tensor != nullptr);
  assert(ends_as_constant_tensor != nullptr);
  assert(axes_as_constant_tensor != nullptr);
  assert(steps_as_constant_tensor != nullptr);
#endif
  const Tensor &begins_tensor = begins_as_constant_tensor->GetValue(),
               &ends_tensor = ends_as_constant_tensor->GetValue(),
               &axes_tensor = axes_as_constant_tensor->GetValue(),
               &steps_tensor = steps_as_constant_tensor->GetValue();
  const std::vector<int64_t> &begins_vector = begins_tensor.GetShape(),
                             &ends_vector = ends_tensor.GetShape(),
                             &axes_vector = axes_tensor.GetShape(),
                             &steps_vector = steps_tensor.GetShape();
#ifdef DEBUG
  assert(begins_vector.size() == 1);
  assert(ends_vector.size() == 1);
  assert(axes_vector.size() == 1);
  assert(steps_vector.size() == 1);
#endif
  std::vector<int64_t> begins_vec, ends_vec, axes_vec, steps_vec;
  const size_t begins_size = begins_vector[0], ends_size = ends_vector[0],
               axes_size = axes_vector[0], steps_size = steps_vector[0];
  for (size_t i = 0; i < begins_size; ++i) {
    begins_vec.push_back(begins_tensor.Get({i}));
  }
  for (size_t i = 0; i < ends_size; ++i) {
    ends_vec.push_back(ends_tensor.Get({i}));
  }
  for (size_t i = 0; i < axes_size; ++i) {
    axes_vec.push_back(axes_tensor.Get({i}));
  }
  for (size_t i = 0; i < steps_size; ++i) {
    steps_vec.push_back(steps_tensor.Get({i}));
  }
  const std::string &input_name = input_as_non_constant->GetName(),
                    &output_name = output_as_non_constant->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
  ptr = std::make_shared<flow::SliceNode>(
      std::move(name), std::move(begins_vec), std::move(ends_vec),
      std::move(axes_vec), std::move(steps_vec), std::move(input_region),
      std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertSoftmaxNode(flow::Flow &flow,
                                       const graph::Graph &graph,
                                       const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Softmax);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
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
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
  std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
  const Meta &input_meta = input_region->GetMeta();
  const Meta &output_meta = output_region->GetMeta();
  assert(input_meta == output_meta);
#endif
  ptr = std::make_shared<flow::SoftmaxNode>(
      std::move(name), std::move(input_region), std::move(output_region), axis);
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertSqrtNode(flow::Flow &flow, const graph::Graph &graph,
                                    const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Sqrt);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::SqrtNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &input_name = input_as_non_constant->GetName(),
                    &output_name = output_as_non_constant->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
  const Meta &input_meta = input_region->GetMeta();
  const Meta &output_meta = output_region->GetMeta();
  assert(input_meta == output_meta);
#endif
  ptr = std::make_shared<flow::SqrtNode>(
      std::move(name), std::move(input_region), std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertSqueezeNode(flow::Flow &flow,
                                       const graph::Graph &graph,
                                       const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Squeeze);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
  std::vector<int64_t> axes;
  if (inputs.size() == 2) {
    std::shared_ptr<graph::Edge> axes_edge = inputs[1];
    std::shared_ptr<graph::ConstantTensorEdge> axes_as_constant_tensor =
        std::dynamic_pointer_cast<graph::ConstantTensorEdge>(axes_edge);
#ifdef DEBUG
    assert(axes_as_constant_tensor != nullptr);
#endif
    Tensor axes_tensor = axes_as_constant_tensor->GetValue();
    const std::vector<int64_t> &axes_shape = axes_tensor.GetShape();
#ifdef DEBUG
    assert(axes_tensor.GetType() == Type::kInt64);
    assert(axes_shape.size() == 1);
#endif
    const size_t size = axes_shape[0];
    for (size_t i = 0; i < size; ++i) {
      axes.push_back(static_cast<int64_t>(axes_tensor.Get({i})));
    }
  } else if (inputs.size() == 1) {
    axes = node.GetAttribute("axes").GetInt64Array();
  }
#ifdef DEBUG
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> input = inputs[0], output = outputs[0];
#ifdef DEBUG
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::SqueezeNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge>
      input_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(input),
      output_as_non_constant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(input_as_non_constant != nullptr);
  assert(output_as_non_constant != nullptr);
#endif
  const std::string &input_name = input->GetName(),
                    &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
#endif
  ptr = std::make_shared<flow::SqueezeNode>(std::move(name), std::move(axes),
                                            std::move(input_region),
                                            std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertSubNode(flow::Flow &flow, const graph::Graph &graph,
                                   const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Sub);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 2);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> lhs = inputs[0], rhs = inputs[1],
                               output = outputs[0];
#ifdef DEBUG
  assert(lhs != nullptr);
  assert(rhs != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::SubNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> output_as_non_constant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  if (std::shared_ptr<graph::ConstantScalarEdge> lhs_as_constant_scalar =
          std::dynamic_pointer_cast<graph::ConstantScalarEdge>(lhs)) {
    if (std::shared_ptr<graph::NonConstantEdge> rhs_as_non_constant =
            std::dynamic_pointer_cast<graph::NonConstantEdge>(rhs)) {
      const std::string &input_name = rhs_as_non_constant->GetName(),
                        &output_name = output->GetName();
      std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name),
                                    output_region = flow.GetRegion(output_name);
#ifdef DEBUG
      assert(input_region != nullptr);
      assert(output_region != nullptr);
      const Meta &input_meta = input_region->GetMeta(),
                 &output_meta = output_region->GetMeta();
      assert(input_meta == output_meta);
#endif
      ptr = std::make_shared<flow::SubConstantLhsNode>(
          std::move(name), lhs_as_constant_scalar->GetType(),
          lhs_as_constant_scalar->GetValue(), std::move(input_region),
          std::move(output_region));
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
  } else if (std::shared_ptr<graph::NonConstantEdge> lhs_as_non_constant =
                 std::dynamic_pointer_cast<graph::NonConstantEdge>(lhs)) {
    if (std::shared_ptr<graph::NonConstantEdge> rhs_as_non_constant =
            std::dynamic_pointer_cast<graph::NonConstantEdge>(rhs)) {
      const std::string &lhs_name = lhs_as_non_constant->GetName(),
                        &rhs_name = rhs_as_non_constant->GetName(),
                        &output_name = output->GetName();
      std::shared_ptr<flow::Region> lhs_region = flow.GetRegion(lhs_name),
                                    rhs_region = flow.GetRegion(rhs_name),
                                    output_region = flow.GetRegion(output_name);
#ifdef DEBUG
      assert(lhs_region != nullptr);
      assert(rhs_region != nullptr);
      assert(output_region != nullptr);
#endif
      ptr = std::make_shared<flow::SubCommonNode>(
          std::move(name), std::move(lhs_region), std::move(rhs_region),
          std::move(output_region));
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertTanhNode(flow::Flow &flow, const graph::Graph &graph,
                                    const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Tanh);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
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
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
  std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
  const Meta &input_meta = input_region->GetMeta();
  const Meta &output_meta = output_region->GetMeta();
  assert(input_meta == output_meta);
#endif
  ptr = std::make_shared<flow::TanhNode>(
      std::move(name), std::move(input_region), std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertTransposeNode(flow::Flow &flow,
                                         const graph::Graph &graph,
                                         const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Transpose);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
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
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
  std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
#endif
  const Meta &input_meta = input_region->GetMeta();
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
      std::move(name), std::move(permutation), std::move(input_region),
      std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertUnsqueezeNode(flow::Flow &flow,
                                         const graph::Graph &graph,
                                         const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Unsqueeze);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
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
  Tensor axes_tensor = axes_as_constant_tensor->GetValue();
  const std::vector<int64_t> &axes_vector = axes_tensor.GetShape();
  const std::string &input_name = input->GetName();
  const std::string &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
  std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
  assert(axes_vector.size() == 1);
#endif
  size_t size = axes_vector[0];
  std::vector<int64_t> axes_data(size, 0);
  for (size_t i = 0; i < size; ++i) {
    axes_data[i] = axes_tensor.Get({i});
  }
  ptr = std::make_shared<flow::UnsqueezeNode>(
      std::move(name), std::move(axes_data), std::move(input_region),
      std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertUnsqueezeSubMulNode(flow::Flow &flow,
                                               const graph::Graph &graph,
                                               const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::UnsqueezeSubMul);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
#ifdef DEBUG
  assert(inputs.size() == 4);
  assert(outputs.size() == 1);
#endif
  std::shared_ptr<graph::Edge> axes = inputs[0];
  std::shared_ptr<graph::Edge> sub = inputs[1];
  std::shared_ptr<graph::Edge> mul = inputs[2];
  std::shared_ptr<graph::Edge> input = inputs[3];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(axes != nullptr);
  assert(sub != nullptr);
  assert(mul != nullptr);
  assert(input != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::UnsqueezeSubLhsScalarMulRhsScalarNode> ptr = nullptr;
  std::shared_ptr<graph::ConstantTensorEdge> axes_as_constant_tensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(axes);
  std::shared_ptr<graph::ConstantScalarEdge> sub_as_constant_scalar =
      std::dynamic_pointer_cast<graph::ConstantScalarEdge>(sub);
  std::shared_ptr<graph::ConstantScalarEdge> mul_as_constant_scalar =
      std::dynamic_pointer_cast<graph::ConstantScalarEdge>(mul);
#ifdef DEBUG
  assert(axes_as_constant_tensor != nullptr);
  assert(sub_as_constant_scalar != nullptr);
  assert(mul_as_constant_scalar != nullptr);
  assert(isa<graph::NonConstantEdge>(input));
  assert(isa<graph::NonConstantEdge>(output));
#endif
  Tensor axes_tensor = axes_as_constant_tensor->GetValue();
  const std::vector<int64_t> &axes_vector = axes_tensor.GetShape();
  const std::string &input_name = input->GetName();
  const std::string &output_name = output->GetName();
  std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
  std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
  assert(input_region != nullptr);
  assert(output_region != nullptr);
  assert(axes_vector.size() == 1);
#endif
  size_t size = axes_vector[0];
  std::vector<int64_t> axes_data(size, 0);
  for (size_t i = 0; i < size; ++i) {
    axes_data[i] = axes_tensor.Get({i});
  }
  ptr = std::make_shared<flow::UnsqueezeSubLhsScalarMulRhsScalarNode>(
      std::move(name), std::move(axes_data), sub_as_constant_scalar->GetType(),
      sub_as_constant_scalar->GetValue(), mul_as_constant_scalar->GetType(),
      mul_as_constant_scalar->GetValue(), std::move(input_region),
      std::move(output_region));
  flow.PutNode(std::move(ptr));
}

void ConverterImpl::convertWhereNode(flow::Flow &flow,
                                     const graph::Graph &graph,
                                     const graph::Node &node) {
#ifdef DEBUG
  assert(node.GetOp() == graph::Node::Op::Where);
#endif
  std::string name = node.GetName();
  std::vector<std::shared_ptr<graph::Edge>> inputs = graph.GetNodeFrom(node),
                                            outputs = graph.GetNodeTo(node);
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
    const std::string &input_name = x->GetName();
    const std::string &output_name = output->GetName();
    std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
    std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
    assert(input_region != nullptr);
    assert(output_region != nullptr);
#endif
    ptr = std::make_shared<flow::WhereConstantCondConstantScalarYNode>(
        std::move(name), std::move(condition_tensor),
        y_as_constant_scalar->GetType(), y_as_constant_scalar->GetValue(),
        std::move(input_region), std::move(output_region));
  } else if (std::shared_ptr<graph::ConstantTensorEdge> y_as_constant_tensor =
                 std::dynamic_pointer_cast<graph::ConstantTensorEdge>(y)) {
    Tensor condition_tensor = condition_as_constant_tensor->GetValue();
    Tensor y_tensor = y_as_constant_tensor->GetValue();
    const std::string &input_name = x->GetName();
    const std::string &output_name = output->GetName();
    std::shared_ptr<flow::Region> input_region = flow.GetRegion(input_name);
    std::shared_ptr<flow::Region> output_region = flow.GetRegion(output_name);
#ifdef DEBUG
    assert(input_region != nullptr);
    assert(output_region != nullptr);
#endif
    ptr = std::make_shared<flow::WhereConstantCondConstantTensorYNode>(
        std::move(name), std::move(condition_tensor), std::move(y_tensor),
        std::move(input_region), std::move(output_region));
  } else {
#ifdef DEBUG
    assert(false && "unreachable");
#else
    __builtin_unreachable();
#endif
  }
  flow.PutNode(std::move(ptr));
}

} // namespace worker
} // namespace fluidml
