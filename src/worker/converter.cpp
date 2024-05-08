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
    if (std::shared_ptr<graph::InputEdge> inputEdge =
            std::dynamic_pointer_cast<graph::InputEdge>(edge)) {
      std::string name = inputEdge->GetName();
      std::vector<std::shared_ptr<graph::Node>> tos =
          graph.GetEdgeTo(*inputEdge);
#ifdef DEBUG
      assert(tos.size() == 1);
#endif
      std::shared_ptr<graph::Node> to = tos[0];
      std::string toName = to->GetName();
      Meta meta = inputEdge->GetMeta();
      name = inputEdge->GetName();
      std::string fromName = name;
      std::shared_ptr<flow::InputRegion> region =
          std::make_shared<flow::InputRegion>(std::move(name), std::move(meta));
      std::shared_ptr<flow::InputRegion> regionClone = region;
      std::shared_ptr<flow::InputEdge> ptr = std::make_shared<flow::InputEdge>(
          std::move(regionClone), std::move(toName));
      flow.PutEdge(std::move(ptr));
      flow.PutRegion(std::move(region));
    } else if (std::shared_ptr<graph::OutputEdge> outputEdge =
                   std::dynamic_pointer_cast<graph::OutputEdge>(edge)) {
      std::string name = outputEdge->GetName();
      std::shared_ptr<graph::Node> from = graph.GetEdgeFrom(*outputEdge);
      std::string fromName = from->GetName();
      Meta meta = outputEdge->GetMeta();
      name = outputEdge->GetName();
      std::string toName = name;
      std::shared_ptr<flow::OutputRegion> region =
          std::make_shared<flow::OutputRegion>(std::move(name),
                                               std::move(meta));
      std::shared_ptr<flow::OutputRegion> regionClone = region;
      std::shared_ptr<flow::OutputEdge> outputPtr =
          std::make_shared<flow::OutputEdge>(std::move(regionClone),
                                             std::move(fromName));
      flow.PutEdge(std::move(outputPtr));
      regionClone = region;
      flow.PutRegion(std::move(regionClone));
      std::vector<std::shared_ptr<graph::Node>> tos =
          graph.GetEdgeTo(*outputEdge);
      for (std::shared_ptr<graph::Node> to : tos) {
        fromName = from->GetName();
        toName = to->GetName();
        regionClone = region;
        std::shared_ptr<flow::MemoryEdge> memoryPtr =
            std::make_shared<flow::MemoryEdge>(
                std::move(regionClone), std::move(fromName), std::move(toName));
        flow.PutEdge(std::move(memoryPtr));
      }
    } else if (std::shared_ptr<graph::PureEdge> pureEdge =
                   std::dynamic_pointer_cast<graph::PureEdge>(edge)) {
      std::string name = pureEdge->GetName();
      Meta meta = pureEdge->GetMeta();
      std::shared_ptr<flow::InnerRegion> region =
          std::make_shared<flow::InnerRegion>(std::move(name), std::move(meta));
      std::shared_ptr<flow::InnerRegion> regionClone = region;
      flow.PutRegion(std::move(regionClone));
      std::shared_ptr<graph::Node> from = graph.GetEdgeFrom(*pureEdge);
      std::vector<std::shared_ptr<graph::Node>> tos =
          graph.GetEdgeTo(*pureEdge);
      for (const std::shared_ptr<graph::Node> &to : tos) {
        std::string fromName = from->GetName();
        std::string toName = to->GetName();
        regionClone = region;
        std::shared_ptr<flow::MemoryEdge> ptr =
            std::make_shared<flow::MemoryEdge>(
                std::move(regionClone), std::move(fromName), std::move(toName));
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
  std::shared_ptr<graph::Edge> inputLhs = inputs[0];
  std::shared_ptr<graph::Edge> inputRhs = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(inputLhs != nullptr);
  assert(inputRhs != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::AddNode> ptr = nullptr;
  std::shared_ptr<graph::ConstantEdge> inputLhsAsConstant =
      std::dynamic_pointer_cast<graph::ConstantEdge>(inputLhs);
  std::shared_ptr<graph::ConstantEdge> inputRhsAsConstant =
      std::dynamic_pointer_cast<graph::ConstantEdge>(inputRhs);
  std::shared_ptr<graph::NonConstantEdge> outputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(outputAsNonConstant != nullptr);
#endif
  Meta outputMeta = outputAsNonConstant->GetMeta();
  if (inputLhsAsConstant == nullptr && inputRhsAsConstant == nullptr) {
    std::shared_ptr<graph::NonConstantEdge> inputLhsAsNonConstant =
        std::dynamic_pointer_cast<graph::NonConstantEdge>(inputLhs);
    std::shared_ptr<graph::NonConstantEdge> inputRhsAsNonConstant =
        std::dynamic_pointer_cast<graph::NonConstantEdge>(inputRhs);
#ifdef DEBUG
    assert(inputLhsAsNonConstant != nullptr);
    assert(inputRhsAsNonConstant != nullptr);
#endif
    const std::string &inputLhsName = inputLhsAsNonConstant->GetName();
    const std::string &inputRhsName = inputRhsAsNonConstant->GetName();
    const std::string &outputName = output->GetName();
    std::shared_ptr<flow::Edge> inputLhsPtr = flow.GetEdge(inputLhsName);
    std::shared_ptr<flow::Edge> inputRhsPtr = flow.GetEdge(inputRhsName);
    std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
    assert(inputLhsPtr != nullptr);
    assert(inputRhsPtr != nullptr);
    assert(outputPtr != nullptr);
    Meta inputLhsMeta = inputLhsPtr->GetMeta();
    Meta inputRhsMeta = inputRhsPtr->GetMeta();
    std::optional<Meta> outputMetaOpt =
        BroadcastShape(inputLhsMeta, inputRhsMeta, outputMeta.GetType());
    assert(outputMetaOpt.has_value());
    assert(*outputMetaOpt == outputMeta);
#endif
    ptr = std::make_shared<flow::AddCommonNode>(
        std::move(name), std::move(inputLhsPtr), std::move(inputRhsPtr),
        std::move(outputPtr));
  }
#ifdef DEBUG
  else if (inputLhsAsConstant != nullptr && inputRhsAsConstant != nullptr) {
    throw UnreachableException();
  }
#endif
  else {
    std::shared_ptr<graph::ConstantEdge> inputConstant = nullptr;
    std::shared_ptr<graph::NonConstantEdge> inputNonConstant = nullptr;
    if (inputLhsAsConstant != nullptr && inputRhsAsConstant == nullptr) {
      inputConstant = inputLhsAsConstant;
      inputNonConstant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(inputRhs);
    } else if (inputLhsAsConstant == nullptr && inputRhsAsConstant != nullptr) {
      inputConstant = inputRhsAsConstant;
      inputNonConstant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(inputLhs);
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
#ifdef DEBUG
    assert(inputConstant != nullptr);
    assert(inputNonConstant != nullptr);
#endif
    if (std::shared_ptr<graph::ConstantScalarEdge> inputConstantScalar =
            std::dynamic_pointer_cast<graph::ConstantScalarEdge>(
                inputConstant)) {
      const std::string &inputName = inputNonConstant->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
      assert(inputPtr != nullptr);
      assert(outputPtr != nullptr);
#endif
      ptr = std::make_shared<flow::AddConstantScalarNode>(
          std::move(name), inputConstantScalar->GetType(),
          inputConstantScalar->GetValue(), std::move(inputPtr),
          std::move(outputPtr));
    } else if (std::shared_ptr<graph::ConstantTensorEdge> inputConstantTensor =
                   std::dynamic_pointer_cast<graph::ConstantTensorEdge>(
                       inputConstant)) {
      Tensor tensor = inputConstantTensor->GetValue();
      const std::string &inputName = inputNonConstant->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
      assert(inputPtr != nullptr);
      assert(outputPtr != nullptr);
#endif
      ptr = std::make_shared<flow::AddConstantTensorNode>(
          std::move(name), std::move(tensor), std::move(inputPtr),
          std::move(outputPtr));
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
  std::shared_ptr<graph::Edge> inputLhs = inputs[0];
  std::shared_ptr<graph::Edge> inputRhs = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(inputLhs != nullptr);
  assert(inputRhs != nullptr);
#endif
  std::shared_ptr<flow::DivNode> ptr = nullptr;
  // The support format of Div operator is limited to the following:
  // the left operator is a tensor, and it's a non-constant edge
  // the right operator is a scalar, and it's a constant scalar edge
  // If new formats occur, the code should be updated.
  std::shared_ptr<graph::ConstantScalarEdge> inputRhsAsConstantScalar =
      std::dynamic_pointer_cast<graph::ConstantScalarEdge>(inputRhs);
#ifdef DEBUG
  assert(isa<graph::NonConstantEdge>(inputLhs));
  assert(inputRhsAsConstantScalar != nullptr);
  assert(isa<graph::NonConstantEdge>(output));
#endif
  const std::string &inputName = inputLhs->GetName();
  const std::string &outputName = output->GetName();
  std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
  std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
  assert(inputPtr != nullptr);
  assert(outputPtr != nullptr);
  Meta meta = inputPtr->GetMeta();
  assert(meta == outputPtr->GetMeta());
#endif
  ptr = std::make_shared<flow::DivConstantScalarNode>(
      std::move(name), inputRhsAsConstantScalar->GetType(),
      inputRhsAsConstantScalar->GetValue(), std::move(inputPtr),
      std::move(outputPtr));
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
  const std::string &inputName = input->GetName();
  const std::string &outputName = output->GetName();
  std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
  std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
  assert(inputPtr != nullptr);
  assert(outputPtr != nullptr);
  Meta inputMeta = inputPtr->GetMeta();
  const Meta &outputMeta = outputPtr->GetMeta();
  assert(inputMeta == outputMeta);
#endif
  ptr = std::make_shared<flow::ErfNode>(std::move(name), std::move(inputPtr),
                                        std::move(outputPtr));
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
  std::shared_ptr<graph::Edge> inputLhs = inputs[0];
  std::shared_ptr<graph::Edge> inputRhs = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
  size_t axis = 0;
  if (node.HasAttribute(flow::GatherNode::kAxisAttrName)) {
    axis = node.GetAttribute(flow::GatherNode::kAxisAttrName).GetInt64();
  }
#ifdef DEBUG
  assert(inputLhs != nullptr);
  assert(inputRhs != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::GatherNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> outputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  if (std::shared_ptr<graph::NonConstantEdge> inputLhsAsNonConstant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(inputLhs)) {
    if (std::shared_ptr<graph::ConstantScalarEdge> inputRhsAsConstantScalar =
            std::dynamic_pointer_cast<graph::ConstantScalarEdge>(inputRhs)) {
      const std::string &inputName = inputLhsAsNonConstant->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
      assert(inputPtr != nullptr);
      assert(outputPtr != nullptr);
#endif
      ptr = std::make_shared<flow::GatherConstantIndexScalarNode>(
          std::move(name), std::move(inputPtr),
          std::lround(inputRhsAsConstantScalar->GetValue()),
          std::move(outputPtr), axis);
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  } else if (std::shared_ptr<graph::ConstantTensorEdge>
                 inputLhsAsConstantTensor =
                     std::dynamic_pointer_cast<graph::ConstantTensorEdge>(
                         inputLhs)) {
    if (std::shared_ptr<graph::NonConstantEdge> inputRhsAsNonConstant =
            std::dynamic_pointer_cast<graph::NonConstantEdge>(inputRhs)) {
      Tensor tensor = inputLhsAsConstantTensor->GetValue();
      const std::string &inputName = inputRhsAsNonConstant->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
      assert(inputPtr != nullptr);
      assert(outputPtr != nullptr);
#endif
      ptr = std::make_shared<flow::GatherConstantDataTensorNode>(
          std::move(name), std::move(tensor), std::move(inputPtr),
          std::move(outputPtr), axis);
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
  std::shared_ptr<graph::NonConstantEdge> inputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(input);
  std::shared_ptr<graph::ConstantTensorEdge> weightsAsConstantTensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(weights);
  std::shared_ptr<graph::ConstantTensorEdge> biasAsConstantTensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(bias);
  std::shared_ptr<graph::NonConstantEdge> outputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(inputAsNonConstant != nullptr);
  assert(weightsAsConstantTensor != nullptr);
  assert(biasAsConstantTensor != nullptr);
  assert(outputAsNonConstant != nullptr);
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
  const std::string &inputName = inputAsNonConstant->GetName();
  const std::string &outputName = outputAsNonConstant->GetName();
  std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
  std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
  Tensor weightsTensor = weightsAsConstantTensor->GetValue();
  Tensor biasTensor = biasAsConstantTensor->GetValue();
  ptr = std::make_shared<flow::GemmConstantWeightsBiasNode>(
      std::move(name), std::move(inputPtr), std::move(weightsTensor),
      std::move(biasTensor), std::move(outputPtr), alpha, beta, transA, transB);
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
  std::shared_ptr<graph::NonConstantEdge> inputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(input);
  std::shared_ptr<graph::ConstantTensorEdge> scaleAsConstantTensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(scale);
  std::shared_ptr<graph::ConstantTensorEdge> biasAsConstantTensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(bias);
  std::shared_ptr<graph::NonConstantEdge> outputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(inputAsNonConstant != nullptr);
  assert(scaleAsConstantTensor != nullptr);
  assert(biasAsConstantTensor != nullptr);
  assert(outputAsNonConstant != nullptr);
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
  const std::string &inputName = inputAsNonConstant->GetName();
  const std::string &outputName = outputAsNonConstant->GetName();
  std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
  std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
  assert(inputPtr != nullptr);
  assert(outputPtr != nullptr);
#endif
  Tensor scaleTensor = scaleAsConstantTensor->GetValue();
  Tensor biasTensor = biasAsConstantTensor->GetValue();
  ptr = std::make_shared<flow::LayerNormalizationConstantScaleBiasNode>(
      std::move(name), std::move(inputPtr), std::move(scaleTensor),
      std::move(biasTensor), std::move(outputPtr), axis, epsilon);
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
  std::shared_ptr<graph::NonConstantEdge> outputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(outputAsNonConstant != nullptr);
#endif
  std::shared_ptr<flow::MatMulNode> ptr = nullptr;
  if (std::shared_ptr<graph::ConstantTensorEdge> lhsAsConstantEdge =
          std::dynamic_pointer_cast<graph::ConstantTensorEdge>(lhs)) {
    if (std::shared_ptr<graph::ConstantTensorEdge> rhsAsConstantEdge =
            std::dynamic_pointer_cast<graph::ConstantTensorEdge>(rhs)) {
// TODO: Currently, this code isn't expected to be reached, because such a case
// is expected to be optimized by the ONNX simplifier.
#ifdef DEBUG
      throw UnimplementedException();
#else
      __builtin_unreachable();
#endif
    } else if (std::shared_ptr<graph::NonConstantEdge> rhsAsNonConstantEdge =
                   std::dynamic_pointer_cast<graph::NonConstantEdge>(rhs)) {
      Tensor lhsTensor = lhsAsConstantEdge->GetValue();
      const std::string &rhsName = rhsAsNonConstantEdge->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> rhsPtr = flow.GetEdge(rhsName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
      assert(rhsPtr != nullptr);
      assert(outputPtr != nullptr);
#endif
      ptr = std::make_shared<flow::MatMulConstantLhsNode>(
          std::move(name), std::move(lhsTensor), std::move(rhsPtr),
          std::move(outputPtr));
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  } else if (std::shared_ptr<graph::NonConstantEdge> lhsAsNonConstantEdge =
                 std::dynamic_pointer_cast<graph::NonConstantEdge>(lhs)) {
    if (std::shared_ptr<graph::ConstantTensorEdge> rhsAsConstantEdge =
            std::dynamic_pointer_cast<graph::ConstantTensorEdge>(rhs)) {
      Tensor rhsTensor = rhsAsConstantEdge->GetValue();
      const std::string &lhsName = lhsAsNonConstantEdge->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> lhsPtr = flow.GetEdge(lhsName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
      assert(lhsPtr != nullptr);
      assert(outputPtr != nullptr);
#endif
      ptr = std::make_shared<flow::MatMulConstantRhsNode>(
          std::move(name), std::move(lhsPtr), std::move(rhsTensor),
          std::move(outputPtr));
    } else if (std::shared_ptr<graph::NonConstantEdge> rhsAsNonConstantEdge =
                   std::dynamic_pointer_cast<graph::NonConstantEdge>(rhs)) {
      const std::string &lhsName = lhsAsNonConstantEdge->GetName();
      const std::string &rhsName = rhsAsNonConstantEdge->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> lhsPtr = flow.GetEdge(lhsName);
      std::shared_ptr<flow::Edge> rhsPtr = flow.GetEdge(rhsName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
      assert(lhsPtr != nullptr);
      assert(rhsPtr != nullptr);
      assert(outputPtr != nullptr);
#endif
      ptr = std::make_shared<flow::MatMulCommonNode>(
          std::move(name), std::move(lhsPtr), std::move(rhsPtr),
          std::move(outputPtr));
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
  std::shared_ptr<graph::Edge> inputLhs = inputs[0];
  std::shared_ptr<graph::Edge> inputRhs = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(inputLhs != nullptr);
  assert(inputRhs != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::MulNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> outputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  if (std::shared_ptr<graph::NonConstantEdge> inputLhsAsNonConstant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(inputLhs)) {
    if (std::shared_ptr<graph::NonConstantEdge> inputRhsAsNonConstant =
            std::dynamic_pointer_cast<graph::NonConstantEdge>(inputRhs)) {
      const std::string &inputLhsName = inputLhsAsNonConstant->GetName();
      const std::string &inputRhsName = inputRhsAsNonConstant->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> inputLhsPtr = flow.GetEdge(inputLhsName);
      std::shared_ptr<flow::Edge> inputRhsPtr = flow.GetEdge(inputRhsName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
      assert(inputLhsPtr != nullptr);
      assert(inputRhsPtr != nullptr);
      assert(outputPtr != nullptr);
#endif
      ptr = std::make_shared<flow::MulCommonNode>(
          std::move(name), std::move(inputLhsPtr), std::move(inputRhsPtr),
          std::move(outputPtr));
    } else if (std::shared_ptr<graph::ConstantScalarEdge>
                   inputRhsAsConstantScalar =
                       std::dynamic_pointer_cast<graph::ConstantScalarEdge>(
                           inputRhs)) {
      const std::string &inputName = inputLhsAsNonConstant->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
      assert(inputPtr != nullptr);
      assert(outputPtr != nullptr);
#endif
      ptr = std::make_shared<flow::MulConstantScalarNode>(
          std::move(name), std::move(inputPtr),
          inputRhsAsConstantScalar->GetType(),
          inputRhsAsConstantScalar->GetValue(), std::move(outputPtr));
    } else if (std::shared_ptr<graph::ConstantTensorEdge>
                   inputRhsAsConstantTensor =
                       std::dynamic_pointer_cast<graph::ConstantTensorEdge>(
                           inputRhs)) {
      Tensor tensor = inputRhsAsConstantTensor->GetValue();
      const std::string &inputName = inputLhsAsNonConstant->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
      assert(inputPtr != nullptr);
      assert(outputPtr != nullptr);
#endif
      ptr = std::make_shared<flow::MulConstantTensorNode>(
          std::move(name), std::move(inputPtr), std::move(tensor),
          std::move(outputPtr));
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  } else if (std::shared_ptr<graph::ConstantTensorEdge> inputLhsAsConstant =
                 std::dynamic_pointer_cast<graph::ConstantTensorEdge>(
                     inputLhs)) {
    if (std::shared_ptr<graph::NonConstantEdge> inputRhsAsNonConstant =
            std::dynamic_pointer_cast<graph::NonConstantEdge>(inputRhs)) {
      Tensor tensor = inputLhsAsConstant->GetValue();
      const std::string &inputName = inputRhsAsNonConstant->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
      assert(inputPtr != nullptr);
      assert(outputPtr != nullptr);
#endif
      ptr = std::make_shared<flow::MulConstantTensorNode>(
          std::move(name), std::move(inputPtr), std::move(tensor),
          std::move(outputPtr));
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  } else if (std::shared_ptr<graph::ConstantScalarEdge> inputLhsAsConstant =
                 std::dynamic_pointer_cast<graph::ConstantScalarEdge>(
                     inputLhs)) {
    if (std::shared_ptr<graph::NonConstantEdge> inputRhsAsNonConstant =
            std::dynamic_pointer_cast<graph::NonConstantEdge>(inputRhs)) {
      const std::string &inputName = inputRhsAsNonConstant->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
      ptr = std::make_shared<flow::MulConstantScalarNode>(
          std::move(name), std::move(inputPtr), inputLhsAsConstant->GetType(),
          inputLhsAsConstant->GetValue(), std::move(outputPtr));
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
  std::shared_ptr<graph::Edge> inputBase = inputs[0];
  std::shared_ptr<graph::Edge> inputExponent = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(inputBase != nullptr);
  assert(inputExponent != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::PowNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> outputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  if (std::shared_ptr<graph::NonConstantEdge> inputBaseAsNonConstant =
          std::dynamic_pointer_cast<graph::NonConstantEdge>(inputBase)) {
    if (std::shared_ptr<graph::ConstantScalarEdge>
            inputExponentAsConstantScalar =
                std::dynamic_pointer_cast<graph::ConstantScalarEdge>(
                    inputExponent)) {
      const std::string &inputName = inputBaseAsNonConstant->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
      assert(inputPtr != nullptr);
      assert(outputPtr != nullptr);
#endif
      ptr = std::make_shared<flow::PowNode>(
          std::move(name), std::move(inputPtr),
          inputExponentAsConstantScalar->GetType(),
          inputExponentAsConstantScalar->GetValue(), std::move(outputPtr));
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
  std::shared_ptr<graph::ConstantTensorEdge> shapeAsConstantTensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(shape);
#ifdef DEBUG
  assert(isa<graph::NonConstantEdge>(input));
  assert(shapeAsConstantTensor != nullptr);
  assert(isa<graph::NonConstantEdge>(output));
#endif
  Tensor shapeTensor = shapeAsConstantTensor->GetValue();
  const std::string &inputName = input->GetName();
  const std::string &outputName = output->GetName();
  std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
  std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
  assert(inputPtr != nullptr);
  assert(outputPtr != nullptr);
  const Meta &outputMeta = outputPtr->GetMeta();
  const std::vector<int64_t> &shapeVector = shapeTensor.GetShape();
  const std::vector<int64_t> &outputShape = outputMeta.GetShape();
  assert(shapeVector.size() == 1);
  size_t size = shapeVector[0];
  assert(size == outputShape.size());
  std::vector<int64_t> shapeVec;
  for (size_t i = 0; i < size; ++i) {
    shapeVec.push_back(shapeTensor.Get({i}));
  }
  Meta expectedMeta = Meta(outputMeta.GetType(), std::move(shapeVec));
  std::optional<Meta> inferredShapeOpt = ReshapeShapeInference(
      expectedMeta, std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                    std::multiplies<int64_t>()));
  assert(inferredShapeOpt.has_value());
  assert(outputMeta == *inferredShapeOpt);
#endif
  ptr = std::make_shared<flow::ReshapeNode>(
      std::move(name), std::move(inputPtr), std::move(outputPtr));
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
  std::shared_ptr<graph::NonConstantEdge> inputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(input);
  std::shared_ptr<graph::NonConstantEdge> outputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(inputAsNonConstant != nullptr);
  assert(outputAsNonConstant != nullptr);
#endif
  int64_t axis = flow::SoftmaxNode::kAxis;
  if (node.HasAttribute(flow::SoftmaxNode::kAxisAttrName)) {
    axis = node.GetAttribute(flow::SoftmaxNode::kAxisAttrName).GetInt64();
  }
  const std::string &inputName = inputAsNonConstant->GetName();
  const std::string &outputName = outputAsNonConstant->GetName();
  std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
  std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
  assert(inputPtr != nullptr);
  assert(outputPtr != nullptr);
  const Meta &inputMeta = inputPtr->GetMeta();
  const Meta &outputMeta = outputPtr->GetMeta();
  assert(inputMeta == outputMeta);
#endif
  ptr = std::make_shared<flow::SoftmaxNode>(
      std::move(name), std::move(inputPtr), std::move(outputPtr), axis);
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
  std::shared_ptr<graph::NonConstantEdge> inputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(input);
  std::shared_ptr<graph::ConstantTensorEdge> shapesAsConstantTensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(shapes);
#ifdef DEBUG
  assert(inputAsNonConstant != nullptr);
  assert(shapesAsConstantTensor != nullptr);
#endif
  const std::string &inputName = inputAsNonConstant->GetName();
  std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
  Tensor shapesTensor = shapesAsConstantTensor->GetValue();
  const std::vector<int64_t> &shape = shapesTensor.GetShape();
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
    std::shared_ptr<graph::NonConstantEdge> outputAsNonConstant =
        std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
    assert(outputAsNonConstant != nullptr);
#endif
    const std::string &outputName = outputAsNonConstant->GetName();
    std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
    assert(outputPtr != nullptr);
    const Meta &outputMeta = outputPtr->GetMeta();
    std::vector<int64_t> outputShape = outputMeta.GetShape();
    assert(outputShape[axis] == shapesTensor.Get({i}));
#endif
    outputPtrs.push_back(std::move(outputPtr));
  }
  ptr = std::make_shared<flow::SplitNode>(std::move(name), std::move(inputPtr),
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
  std::shared_ptr<graph::Edge> inputLhs = inputs[0];
  std::shared_ptr<graph::Edge> inputRhs = inputs[1];
  std::shared_ptr<graph::Edge> output = outputs[0];
#ifdef DEBUG
  assert(inputLhs != nullptr);
  assert(inputRhs != nullptr);
  assert(output != nullptr);
#endif
  std::shared_ptr<flow::SubNode> ptr = nullptr;
  std::shared_ptr<graph::NonConstantEdge> outputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
  // The support format of Sub operator is limited to the following: the left
  // operator is a constant scalar, and the right operator is a non-constant
  // edge.
  // If new formats occur, the code should be updated.
  if (std::shared_ptr<graph::ConstantScalarEdge> inputLhsAsConstantScalar =
          std::dynamic_pointer_cast<graph::ConstantScalarEdge>(inputLhs)) {
    if (std::shared_ptr<graph::NonConstantEdge> inputRhsAsNonConstant =
            std::dynamic_pointer_cast<graph::NonConstantEdge>(inputRhs)) {
      const std::string &inputName = inputRhsAsNonConstant->GetName();
      const std::string &outputName = output->GetName();
      std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
      std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
      assert(inputPtr != nullptr);
      assert(outputPtr != nullptr);
      const Meta &inputMeta = inputPtr->GetMeta();
      const Meta &outputMeta = outputPtr->GetMeta();
      assert(inputMeta == outputMeta);
#endif
      ptr = std::make_shared<flow::SubConstantScalarLhsNode>(
          std::move(name), std::move(inputPtr),
          inputLhsAsConstantScalar->GetType(),
          inputLhsAsConstantScalar->GetValue(), std::move(outputPtr));
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
  std::shared_ptr<graph::NonConstantEdge> inputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(input);
  std::shared_ptr<graph::NonConstantEdge> outputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(inputAsNonConstant != nullptr);
  assert(outputAsNonConstant != nullptr);
#endif
  const std::string &inputName = inputAsNonConstant->GetName();
  const std::string &outputName = outputAsNonConstant->GetName();
  std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
  std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
  assert(inputPtr != nullptr);
  assert(outputPtr != nullptr);
  const Meta &inputMeta = inputPtr->GetMeta();
  const Meta &outputMeta = outputPtr->GetMeta();
  assert(inputMeta == outputMeta);
#endif
  ptr = std::make_shared<flow::TanhNode>(std::move(name), std::move(inputPtr),
                                         std::move(outputPtr));
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
  std::shared_ptr<graph::NonConstantEdge> inputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(input);
  std::shared_ptr<graph::NonConstantEdge> outputAsNonConstant =
      std::dynamic_pointer_cast<graph::NonConstantEdge>(output);
#ifdef DEBUG
  assert(inputAsNonConstant != nullptr);
  assert(outputAsNonConstant != nullptr);
#endif
  const std::string &inputName = inputAsNonConstant->GetName();
  const std::string &outputName = outputAsNonConstant->GetName();
  std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
  std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
  assert(inputPtr != nullptr);
  assert(outputPtr != nullptr);
#endif
  const Meta &inputMeta = inputPtr->GetMeta();
  const std::vector<int64_t> &inputShape = inputMeta.GetShape();
  size_t inputShapeLen = inputShape.size();
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
      std::move(name), std::move(inputPtr), std::move(outputPtr),
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
  std::shared_ptr<graph::ConstantTensorEdge> axesAsConstantTensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(axes);
#ifdef DEBUG
  assert(axesAsConstantTensor != nullptr);
  assert(isa<graph::NonConstantEdge>(input));
  assert(isa<graph::NonConstantEdge>(output));
#endif
  Tensor axesTensor = axesAsConstantTensor->GetValue();
  const std::vector<int64_t> &axesVector = axesTensor.GetShape();
  const std::string &inputName = input->GetName();
  const std::string &outputName = output->GetName();
  std::shared_ptr<flow::Edge> inputPtr = flow.GetEdge(inputName);
  std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
  assert(inputPtr != nullptr);
  assert(outputPtr != nullptr);
  assert(axesVector.size() == 1);
#endif
  size_t size = axesVector[0];
  std::vector<int64_t> axesVec(size, 0);
  for (size_t i = 0; i < size; ++i) {
    axesVec[i] = axesTensor.Get({i});
  }
  ptr = std::make_shared<flow::UnsqueezeNode>(
      std::move(name), std::move(inputPtr), std::move(outputPtr),
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
  std::shared_ptr<graph::ConstantTensorEdge> conditionAsConstantTensor =
      std::dynamic_pointer_cast<graph::ConstantTensorEdge>(condition);
#ifdef DEBUG
  assert(conditionAsConstantTensor != nullptr);
  assert(isa<graph::NonConstantEdge>(x));
  assert(isa<graph::NonConstantEdge>(output));
#endif
  if (std::shared_ptr<graph::ConstantScalarEdge> yAsConstantScalar =
          std::dynamic_pointer_cast<graph::ConstantScalarEdge>(y)) {
    Tensor conditionTensor = conditionAsConstantTensor->GetValue();
    const std::string &xName = x->GetName();
    const std::string &outputName = output->GetName();
    std::shared_ptr<flow::Edge> xPtr = flow.GetEdge(xName);
    std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
    assert(xPtr != nullptr);
    assert(outputPtr != nullptr);
#endif
    ptr = std::make_shared<flow::WhereConstantCondConstantScalarYNode>(
        std::move(name), std::move(conditionTensor), std::move(xPtr),
        yAsConstantScalar->GetType(), yAsConstantScalar->GetValue(),
        std::move(outputPtr));
  } else if (std::shared_ptr<graph::ConstantTensorEdge> yAsConstantTensor =
                 std::dynamic_pointer_cast<graph::ConstantTensorEdge>(y)) {
    Tensor conditionTensor = conditionAsConstantTensor->GetValue();
    Tensor yTensor = yAsConstantTensor->GetValue();
    const std::string &xName = x->GetName();
    const std::string &outputName = output->GetName();
    std::shared_ptr<flow::Edge> xPtr = flow.GetEdge(xName);
    std::shared_ptr<flow::Edge> outputPtr = flow.GetEdge(outputName);
#ifdef DEBUG
    assert(xPtr != nullptr);
    assert(outputPtr != nullptr);
#endif
    ptr = std::make_shared<flow::WhereConstantCondConstantTensorYNode>(
        std::move(name), std::move(conditionTensor), std::move(xPtr),
        std::move(yTensor), std::move(outputPtr));
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
