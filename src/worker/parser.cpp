#include "worker/parser.h"
#include "onnx/checker.h"
#include "onnx/common/file_utils.h"
#include "onnx/onnx-ml.pb.h"
#include "structure/graph/attribute.h"
#include "structure/graph/edge.h"
#include "structure/graph/graph.h"
#include "utils/float.h"
#include "utils/type.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>
#ifdef DEBUG
#include "exception/unimplemented_exception.h"
#include "exception/unreachable_exception.h"
#include <cassert>
#endif

namespace {
using namespace cpu_transformers;
using namespace cpu_transformers::graph;
template <typename E,
          typename = std::enable_if_t<std::is_base_of<Edge, E>::value>>
void createEdge(Graph &graph, const onnx::ValueInfoProto &proto) {
  std::string name = proto.name();
  onnx::TypeProto_Tensor tensor_type = proto.type().tensor_type();
  Type type = GetType(tensor_type.elem_type());
  std::vector<int64_t> shape;
#ifdef DEBUG
  assert(tensor_type.has_shape());
#endif
  for (const auto &dim : tensor_type.shape().dim()) {
    shape.push_back(dim.dim_value());
  }
  std::shared_ptr<E> edge =
      std::make_shared<E>(std::move(name), type, std::move(shape));
  graph.AddEdge(std::move(edge));
}

template <typename T>
std::vector<float64_t> getTensorProtoAs(const onnx::TensorProto &tensorProto) {
  std::vector<float64_t> data;
  if (tensorProto.has_raw_data()) {
    const T *raw_data =
        reinterpret_cast<const T *>(tensorProto.raw_data().data());
    for (size_t i = 0; i < tensorProto.raw_data().size() / sizeof(T); ++i) {
      data.push_back(raw_data[i]);
    }
  } else if (tensorProto.has_data_type()) {
    if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int32_t>) {
      for (const T &item : tensorProto.int32_data()) {
        data.push_back(item);
      }
    } else if constexpr (std::is_same_v<T, int64_t>) {
      for (const T &item : tensorProto.int64_data()) {
        data.push_back(item);
      }
    } else if constexpr (std::is_same_v<T, float32_t>) {
      for (const T &item : tensorProto.float_data()) {
        data.push_back(item);
      }
    } else if constexpr (std::is_same_v<T, float64_t>) {
      for (const T &item : tensorProto.double_data()) {
        data.push_back(item);
      }
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
  return data;
}

void createNode(Graph &graph, const onnx::NodeProto &nodeProto, Node::Op op) {
  std::string name = nodeProto.name();
  std::unordered_map<std::string, Attribute> attributes;
  for (const auto &attribute : nodeProto.attribute()) {
    std::string attribute_name = attribute.name();
    if (attribute.type() ==
        onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
      if (op == Node::Op::Cast) {
        attributes.insert(
            {std::move(attribute_name), Attribute(GetType(attribute.i()))});
      } else {
        attributes.insert(
            {std::move(attribute_name), Attribute(attribute.i())});
      }
    } else if (attribute.type() == onnx::AttributeProto_AttributeType::
                                       AttributeProto_AttributeType_FLOAT) {
      attributes.insert({std::move(attribute_name), Attribute(attribute.f())});
    } else if (attribute.type() == onnx::AttributeProto_AttributeType::
                                       AttributeProto_AttributeType_INTS) {
      const google::protobuf::RepeatedField<::google::protobuf::int64> &ints =
          attribute.ints();
      attributes.insert(
          {std::move(attribute_name),
           Attribute(std::vector<int64_t>(ints.begin(), ints.end()))});
    } else if (attribute.type() == onnx::AttributeProto_AttributeType::
                                       AttributeProto_AttributeType_TENSOR) {
      const onnx::TensorProto &tensor = attribute.t();
      Type type = GetType(tensor.data_type());
      std::vector<int64_t> shape;
      for (const auto &dim : tensor.dims()) {
        shape.push_back(dim);
      }
      std::vector<float64_t> data;
      switch (type) {
      case Type::BOOL:
        data = getTensorProtoAs<bool>(tensor);
        break;
      case Type::INT64:
        data = getTensorProtoAs<int64_t>(tensor);
        break;
      case Type::FLOAT16:
      case Type::FLOAT32:
        data = getTensorProtoAs<float32_t>(tensor);
        break;
      case Type::FLOAT64:
        data = getTensorProtoAs<float64_t>(tensor);
        break;
      default:
#ifdef DEBUG
        throw UnimplementedException();
#else
        __builtin_unreachable();
#endif
      }
      Tensor tensor_data = Tensor(type, std::move(shape), std::move(data));
      attributes.insert(
          {std::move(attribute_name), Attribute(std::move(tensor_data))});
    } else {
#ifdef DEBUG
      throw UnimplementedException();
#else
      __builtin_unreachable();
#endif
    }
  }
  std::shared_ptr<Node> node =
      std::make_shared<Node>(std::move(name), op, std::move(attributes));
  graph.PutNode(std::move(node));
  for (const std::string &input : nodeProto.input()) {
    graph.EdgeToNode(input, nodeProto.name());
  }
  for (const std::string &output : nodeProto.output()) {
    graph.NodeToEdge(nodeProto.name(), output);
  }
}
} // namespace

namespace cpu_transformers {
namespace worker {
Graph Parser::Run(const std::string &file) {
  onnx::ModelProto model_proto;
  onnx::LoadProtoFromPath(file, model_proto);
  return Run(model_proto);
}

Graph Parser::Run(onnx::ModelProto &model_proto) {
  Graph graph;
#ifdef DEBUG
  onnx::checker::check_model(model_proto);
#endif
  onnx::GraphProto graph_proto = model_proto.graph();

  for (const onnx::ValueInfoProto &input : graph_proto.input()) {
    createEdge<InputEdge>(graph, input);
  }
  for (const onnx::ValueInfoProto &output : graph_proto.output()) {
    createEdge<OutputEdge>(graph, output);
  }
  for (const onnx::ValueInfoProto &value_info : graph_proto.value_info()) {
    createEdge<PureEdge>(graph, value_info);
  }
  for (const onnx::TensorProto &initializer : graph_proto.initializer()) {
    std::string name = initializer.name();
    Type type = GetType(initializer.data_type());
    std::vector<int64_t> shape;
    for (const auto &dim : initializer.dims()) {
      shape.push_back(dim);
    }
    std::vector<float64_t> data;

    // If the data is short, it's stored in the initializer directly. If its
    // size is too large, it's stored in raw_data.
    if (type == Type::BOOL) {
      data = getTensorProtoAs<bool>(initializer);
    } else if (type == Type::INT64) {
      data = getTensorProtoAs<int64_t>(initializer);
    } else if (type == Type::FLOAT32) {
      data = getTensorProtoAs<float32_t>(initializer);
    } else {
#ifdef DEBUG
      throw UnimplementedException();
#else
      __builtin_unreachable();
#endif
    }

    std::shared_ptr<ConstantEdge> edge = nullptr;
    if (shape.size() > 0) {
      Tensor tensor = Tensor(type, std::move(shape), std::move(data));
      edge = std::make_shared<ConstantTensorEdge>(std::move(name), type,
                                                  std::move(tensor));
    } else {
#ifdef DEBUG
      assert(data.size() == 1);
#endif
      edge =
          std::make_shared<ConstantScalarEdge>(std::move(name), type, data[0]);
    }
    graph.AddEdge(std::move(edge));
  }

  for (const onnx::NodeProto &node : graph_proto.node()) {
    if (node.op_type() == "Add") {
      createNode(graph, node, Node::Op::Add);
    } else if (node.op_type() == "Cast") {
      createNode(graph, node, Node::Op::Cast);
    } else if (node.op_type() == "ConstantOfShape") {
      createNode(graph, node, Node::Op::ConstantOfShape);
    } else if (node.op_type() == "Div") {
      createNode(graph, node, Node::Op::Div);
    } else if (node.op_type() == "Equal") {
      createNode(graph, node, Node::Op::Equal);
    } else if (node.op_type() == "Erf") {
      createNode(graph, node, Node::Op::Erf);
    } else if (node.op_type() == "Gather") {
      createNode(graph, node, Node::Op::Gather);
    } else if (node.op_type() == "Gemm") {
      createNode(graph, node, Node::Op::Gemm);
    } else if (node.op_type() == "LayerNormalization") {
      createNode(graph, node, Node::Op::LayerNormalization);
    } else if (node.op_type() == "MatMul") {
      createNode(graph, node, Node::Op::MatMul);
    } else if (node.op_type() == "Mul") {
      createNode(graph, node, Node::Op::Mul);
    } else if (node.op_type() == "Pow") {
      createNode(graph, node, Node::Op::Pow);
    } else if (node.op_type() == "Reshape") {
      createNode(graph, node, Node::Op::Reshape);
    } else if (node.op_type() == "Softmax") {
      createNode(graph, node, Node::Op::Softmax);
    } else if (node.op_type() == "Split") {
      createNode(graph, node, Node::Op::Split);
    } else if (node.op_type() == "Sub") {
      createNode(graph, node, Node::Op::Sub);
    } else if (node.op_type() == "Tanh") {
      createNode(graph, node, Node::Op::Tanh);
    } else if (node.op_type() == "Transpose") {
      createNode(graph, node, Node::Op::Transpose);
    } else if (node.op_type() == "Unsqueeze") {
      createNode(graph, node, Node::Op::Unsqueeze);
    } else if (node.op_type() == "Where") {
      createNode(graph, node, Node::Op::Where);
    } else {
#ifdef DEBUG
      throw UnimplementedException();
#else
      __builtin_unreachable();
#endif
    }
  }

  return graph;
}
} // namespace worker
} // namespace cpu_transformers
