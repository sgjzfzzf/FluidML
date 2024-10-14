#include "structure/graph/attribute.h"
#include "structure/graph/edge.h"
#include "structure/graph/graph.h"
#include "structure/graph/node.h"
#include "utils/isa.hpp"
#include "worker/parser.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

using namespace cpu_transformers;
using namespace cpu_transformers::graph;
using namespace cpu_transformers::worker;

TEST(NodeTest, AddTest0) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_add0_PATH);
  std::shared_ptr<Node> add = graph.GetNode("add");
  std::shared_ptr<Edge> input0 = graph.GetEdge("input0");
  std::shared_ptr<Edge> input1 = graph.GetEdge("input1");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(add, nullptr);
  ASSERT_EQ(add->GetOp(), Node::Op::Add);
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(input1, nullptr);
  ASSERT_NE(output, nullptr);
  std::vector<std::shared_ptr<Edge>> froms = graph.GetNodeFrom(*add);
  ASSERT_EQ(froms.size(), 2);
  std::vector<std::shared_ptr<Edge>> to = graph.GetNodeTo(*add);
  ASSERT_EQ(to.size(), 1);
  ASSERT_EQ(input0->GetType(), Type::kFloat32);
  const std::vector<int64_t> &input0_shape = input0->GetShape();
  ASSERT_EQ(input0_shape.size(), 3);
  ASSERT_EQ(input0_shape[0], 1);
  ASSERT_EQ(input0_shape[1], 128);
  ASSERT_EQ(input0_shape[2], 768);
  ASSERT_EQ(input1->GetType(), Type::kFloat32);
  const std::vector<int64_t> &input1_shape = input1->GetShape();
  ASSERT_EQ(input1_shape.size(), 3);
  ASSERT_EQ(input1_shape[0], 1);
  ASSERT_EQ(input1_shape[1], 128);
  ASSERT_EQ(input1_shape[2], 768);
  const std::vector<int64_t> &output_shape = output->GetShape();
  ASSERT_EQ(output_shape.size(), 3);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 128);
  ASSERT_EQ(output_shape[2], 768);
}

TEST(NodeTest, AddTest1) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_add1_PATH);
  std::shared_ptr<Node> add = graph.GetNode("add");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> weights = graph.GetEdge("weights");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(add, nullptr);
  ASSERT_EQ(add->GetOp(), Node::Op::Add);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(weights, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<InputEdge>(input));
  ASSERT_TRUE(isa<ConstantTensorEdge>(weights));
  ASSERT_TRUE(isa<OutputEdge>(output));
  std::vector<std::shared_ptr<Edge>> froms = graph.GetNodeFrom(*add);
  ASSERT_EQ(froms.size(), 2);
  std::vector<std::shared_ptr<Edge>> to = graph.GetNodeTo(*add);
  ASSERT_EQ(to.size(), 1);
}

// Note: skip it
TEST(NodeTest, CastTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_cast_PATH);
  std::shared_ptr<Node> cast = graph.GetNode("cast");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(cast, nullptr);
  ASSERT_EQ(cast->GetOp(), Node::Op::Cast);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<InputEdge>(input));
  ASSERT_TRUE(isa<OutputEdge>(output));
  const std::vector<int64_t> &input_shape = input->GetShape();
  const std::vector<int64_t> &output_shape = output->GetShape();
  ASSERT_EQ(input_shape.size(), 4);
  ASSERT_EQ(output_shape.size(), 4);
  ASSERT_EQ(input_shape[0], 1);
  ASSERT_EQ(input_shape[1], 1);
  ASSERT_EQ(input_shape[2], 1);
  ASSERT_EQ(input_shape[3], 128);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 1);
  ASSERT_EQ(output_shape[2], 1);
  ASSERT_EQ(output_shape[3], 128);
  Attribute to = cast->GetAttribute("to");
  ASSERT_EQ(to.GetType(), Attribute::Type::DataType);
  ASSERT_EQ(to.GetDataType(), Type::kFloat32);
  ASSERT_NE(to.GetDataType(), Type::kInt64);
}

TEST(NodeTest, ConstantOfShapeTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_constant_of_shape_PATH);
  std::shared_ptr<Node> constant_of_shape = graph.GetNode("constant_of_shape");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(constant_of_shape, nullptr);
  ASSERT_EQ(constant_of_shape->GetOp(), Node::Op::ConstantOfShape);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<ConstantEdge>(input));
  ASSERT_TRUE(isa<OutputEdge>(output));
  const std::vector<int64_t> &input_shape = input->GetShape();
  const std::vector<int64_t> &output_shape = output->GetShape();
  ASSERT_EQ(input_shape.size(), 1);
  ASSERT_EQ(output_shape.size(), 1);
  ASSERT_EQ(input_shape[0], 1);
  ASSERT_EQ(output_shape[0], 2);
  Attribute value = constant_of_shape->GetAttribute("value");
  ASSERT_EQ(value.GetType(), Attribute::Type::Tensor);
  const Tensor &tensor = value.GetTensor();
  ASSERT_EQ(tensor.GetType(), Type::kInt64);
  ASSERT_EQ(tensor.Get({0}), 1);
}

TEST(NodeTest, DivTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_div_PATH);
  std::shared_ptr<Node> div = graph.GetNode("div");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> weights = graph.GetEdge("weights");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(div, nullptr);
  ASSERT_EQ(div->GetOp(), Node::Op::Div);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(weights, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<InputEdge>(input));
  ASSERT_TRUE(isa<ConstantScalarEdge>(weights));
  ASSERT_TRUE(isa<OutputEdge>(output));
  std::shared_ptr<ConstantScalarEdge> constant_weights =
      std::dynamic_pointer_cast<ConstantScalarEdge>(weights);
  ASSERT_NE(constant_weights, nullptr);
  ASSERT_EQ(constant_weights->GetType(), Type::kFloat32);
  ASSERT_EQ(constant_weights->GetValue(), 2);
}

TEST(NodeTest, EqualTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_equal_PATH);
  std::shared_ptr<Node> equal = graph.GetNode("equal");
  std::shared_ptr<Edge> input0 = graph.GetEdge("input0");
  std::shared_ptr<Edge> input1 = graph.GetEdge("input1");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(equal, nullptr);
  ASSERT_EQ(equal->GetOp(), Node::Op::Equal);
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(input1, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<InputEdge>(input1));
  ASSERT_TRUE(isa<OutputEdge>(output));
  const std::vector<int64_t> &input0_shape = input0->GetShape();
  const std::vector<int64_t> &input1_shape = input1->GetShape();
  const std::vector<int64_t> &output_shape = output->GetShape();
  ASSERT_EQ(input0_shape.size(), 1);
  ASSERT_EQ(input1_shape.size(), 1);
  ASSERT_EQ(output_shape.size(), 1);
  ASSERT_EQ(input0_shape[0], 2);
  ASSERT_EQ(input1_shape[0], 2);
  ASSERT_EQ(output_shape[0], 2);
  std::shared_ptr<ConstantTensorEdge> input0_as_constant_tensor =
      std::dynamic_pointer_cast<ConstantTensorEdge>(input0);
  ASSERT_NE(input0_as_constant_tensor, nullptr);
  const Tensor &input0_tensor = input0_as_constant_tensor->GetValue();
  ASSERT_EQ(input0_tensor.GetType(), Type::kInt64);
  ASSERT_EQ(input0_tensor.Get({0}), 1);
  ASSERT_EQ(input0_tensor.Get({1}), 128);
}

TEST(NodeTest, ErfTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_erf_PATH);
  std::shared_ptr<Node> erf = graph.GetNode("erf");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(erf, nullptr);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(erf->GetOp(), Node::Op::Erf);
  const std::vector<int64_t> &input_shape = input->GetShape();
  const std::vector<int64_t> &output_shape = output->GetShape();
  ASSERT_EQ(input_shape.size(), 3);
  ASSERT_EQ(output_shape.size(), 3);
  ASSERT_EQ(input_shape[0], 1);
  ASSERT_EQ(input_shape[1], 128);
  ASSERT_EQ(input_shape[2], 3072);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 128);
  ASSERT_EQ(output_shape[2], 3072);
}

TEST(NodeTest, GatherTest0) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_gather0_PATH);
  std::shared_ptr<Node> gather = graph.GetNode("gather");
  std::shared_ptr<Edge> data = graph.GetEdge("data");
  std::shared_ptr<Edge> indices = graph.GetEdge("indices");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(gather, nullptr);
  ASSERT_EQ(gather->GetOp(), Node::Op::Gather);
  ASSERT_NE(data, nullptr);
  ASSERT_NE(indices, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<ConstantTensorEdge>(data));
  ASSERT_TRUE(isa<InputEdge>(indices));
  ASSERT_TRUE(isa<OutputEdge>(output));
  std::vector<std::shared_ptr<Edge>> froms = graph.GetNodeFrom(*gather);
  ASSERT_EQ(froms.size(), 2);
  std::vector<std::shared_ptr<Edge>> to = graph.GetNodeTo(*gather);
  ASSERT_EQ(to.size(), 1);
}

TEST(NodeTest, GatherTest1) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_gather1_PATH);
  std::shared_ptr<Node> gather = graph.GetNode("gather");
  std::shared_ptr<Edge> data = graph.GetEdge("data");
  std::shared_ptr<Edge> indices = graph.GetEdge("indices");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(gather, nullptr);
  ASSERT_EQ(gather->GetOp(), Node::Op::Gather);
  ASSERT_NE(data, nullptr);
  ASSERT_NE(indices, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<InputEdge>(data));
  ASSERT_TRUE(isa<ConstantScalarEdge>(indices));
  ASSERT_TRUE(isa<OutputEdge>(output));
  Attribute axis = gather->GetAttribute("axis");
  ASSERT_EQ(axis.GetType(), Attribute::Type::Int64);
  ASSERT_EQ(axis.GetInt64(), 1);
}

TEST(NodeTest, GemmTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_gemm_PATH);
  std::shared_ptr<Node> gemm = graph.GetNode("gemm");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> weights = graph.GetEdge("weights");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(gemm, nullptr);
  ASSERT_EQ(gemm->GetOp(), Node::Op::Gemm);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(weights, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<InputEdge>(input));
  ASSERT_TRUE(isa<ConstantTensorEdge>(weights));
  ASSERT_TRUE(isa<OutputEdge>(output));
  Attribute alpha = gemm->GetAttribute("alpha");
  ASSERT_EQ(alpha.GetType(), Attribute::Type::Float32);
  Attribute beta = gemm->GetAttribute("beta");
  ASSERT_EQ(beta.GetType(), Attribute::Type::Float32);
  Attribute transB = gemm->GetAttribute("transB");
  ASSERT_EQ(transB.GetType(), Attribute::Type::Int64);
}

TEST(NodeTest, LayerNormalizationTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_layer_normalization_PATH);
  std::shared_ptr<Node> layernorm = graph.GetNode("layer_normalization");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> scale = graph.GetEdge("scale");
  std::shared_ptr<Edge> bias = graph.GetEdge("bias");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(layernorm, nullptr);
  ASSERT_EQ(layernorm->GetOp(), Node::Op::LayerNormalization);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(scale, nullptr);
  ASSERT_NE(bias, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<InputEdge>(input));
  ASSERT_TRUE(isa<ConstantTensorEdge>(scale));
  ASSERT_TRUE(isa<ConstantTensorEdge>(bias));
  Attribute axis = layernorm->GetAttribute("axis");
  ASSERT_EQ(axis.GetType(), Attribute::Type::Int64);
  Attribute epsilon = layernorm->GetAttribute("epsilon");
  ASSERT_EQ(epsilon.GetType(), Attribute::Type::Float32);
}

TEST(NodeTest, MatmulTest0) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_matmul0_PATH);
  std::shared_ptr<Node> matmul = graph.GetNode("matmul");
  std::shared_ptr<Edge> input0 = graph.GetEdge("input0");
  std::shared_ptr<Edge> input1 = graph.GetEdge("input1");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(matmul, nullptr);
  ASSERT_EQ(matmul->GetOp(), Node::Op::MatMul);
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(input1, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<InputEdge>(input0));
  ASSERT_TRUE(isa<InputEdge>(input1));
  ASSERT_TRUE(isa<OutputEdge>(output));
  std::vector<std::shared_ptr<Edge>> froms = graph.GetNodeFrom(*matmul);
  ASSERT_EQ(froms.size(), 2);
  std::vector<std::shared_ptr<Edge>> to = graph.GetNodeTo(*matmul);
  ASSERT_EQ(to.size(), 1);
  ASSERT_EQ(output->GetName(), "output");
  ASSERT_EQ(input0->GetType(), Type::kFloat32);
  const std::vector<int64_t> &input0_shape = input0->GetShape();
  ASSERT_EQ(input0_shape.size(), 4);
  ASSERT_EQ(input0_shape[0], 1);
  ASSERT_EQ(input0_shape[1], 12);
  ASSERT_EQ(input0_shape[2], 128);
  ASSERT_EQ(input0_shape[3], 64);
  ASSERT_EQ(input1->GetType(), Type::kFloat32);
  const std::vector<int64_t> &input1_shape = input1->GetShape();
  ASSERT_EQ(input1_shape.size(), 4);
  ASSERT_EQ(input1_shape[0], 1);
  ASSERT_EQ(input1_shape[1], 12);
  ASSERT_EQ(input1_shape[2], 64);
  ASSERT_EQ(input1_shape[3], 128);
  ASSERT_EQ(output->GetType(), Type::kFloat32);
  const std::vector<int64_t> &output_shape = output->GetShape();
  ASSERT_EQ(output_shape.size(), 4);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 12);
  ASSERT_EQ(output_shape[2], 128);
  ASSERT_EQ(output_shape[3], 128);
}

TEST(NodeTest, MatmulTest1) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_matmul1_PATH);
  std::shared_ptr<Node> matmul = graph.GetNode("matmul");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> weights = graph.GetEdge("weights");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(matmul, nullptr);
  ASSERT_EQ(matmul->GetOp(), Node::Op::MatMul);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(weights, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<InputEdge>(input));
  ASSERT_TRUE(isa<ConstantTensorEdge>(weights));
  std::vector<std::shared_ptr<Edge>> froms = graph.GetNodeFrom(*matmul);
  ASSERT_EQ(froms.size(), 2);
  std::vector<std::shared_ptr<Edge>> to = graph.GetNodeTo(*matmul);
  ASSERT_EQ(to.size(), 1);
}

TEST(NodeTest, Mul0Test) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_mul0_PATH);
  std::shared_ptr<Node> mul = graph.GetNode("mul");
  std::shared_ptr<Edge> input0 = graph.GetEdge("input0");
  std::shared_ptr<Edge> input1 = graph.GetEdge("input1");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(mul, nullptr);
  ASSERT_EQ(mul->GetOp(), Node::Op::Mul);
  ASSERT_NE(input0, nullptr);
  ASSERT_NE(input1, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<InputEdge>(input0));
  ASSERT_TRUE(isa<InputEdge>(input1));
  ASSERT_TRUE(isa<OutputEdge>(output));
}

TEST(NodeTest, Mul1Test) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_mul1_PATH);
  std::shared_ptr<Node> mul = graph.GetNode("mul");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> weights = graph.GetEdge("weights");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(mul, nullptr);
  ASSERT_EQ(mul->GetOp(), Node::Op::Mul);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(weights, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<InputEdge>(input));
  ASSERT_TRUE(isa<ConstantScalarEdge>(weights));
  ASSERT_TRUE(isa<OutputEdge>(output));
  std::shared_ptr<ConstantScalarEdge> constant_weights =
      std::dynamic_pointer_cast<ConstantScalarEdge>(weights);
  ASSERT_NE(constant_weights, nullptr);
  ASSERT_EQ(constant_weights->GetType(), Type::kFloat32);
  ASSERT_EQ(constant_weights->GetValue(), 1);
}

TEST(NodeTest, SplitTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_split_PATH);
  std::shared_ptr<Node> split = graph.GetNode("split");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> sizes = graph.GetEdge("sizes");
  std::shared_ptr<Edge> output0 = graph.GetEdge("output0");
  std::shared_ptr<Edge> output1 = graph.GetEdge("output1");
  std::shared_ptr<Edge> output2 = graph.GetEdge("output2");
  ASSERT_NE(split, nullptr);
  ASSERT_EQ(split->GetOp(), Node::Op::Split);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(sizes, nullptr);
  ASSERT_NE(output0, nullptr);
  ASSERT_NE(output1, nullptr);
  ASSERT_NE(output2, nullptr);
  ASSERT_TRUE(isa<InputEdge>(input));
  ASSERT_TRUE(isa<OutputEdge>(output0));
  ASSERT_TRUE(isa<OutputEdge>(output1));
  ASSERT_TRUE(isa<OutputEdge>(output2));
  const std::vector<int64_t> &input_shape = input->GetShape();
  ASSERT_EQ(input_shape.size(), 3);
  ASSERT_EQ(input_shape[0], 1);
  ASSERT_EQ(input_shape[1], 128);
  ASSERT_EQ(input_shape[2], 2304);
  std::shared_ptr<ConstantTensorEdge> constantSizes =
      std::dynamic_pointer_cast<ConstantTensorEdge>(sizes);
  ASSERT_NE(constantSizes, nullptr);
  const Tensor &sizes_tensor = constantSizes->GetValue();
  ASSERT_EQ(sizes_tensor.GetType(), Type::kInt64);
  ASSERT_EQ(sizes_tensor.Get({0}), 768);
  ASSERT_EQ(sizes_tensor.Get({1}), 768);
  ASSERT_EQ(sizes_tensor.Get({2}), 768);
  const std::vector<int64_t> &output0_shape = output0->GetShape();
  ASSERT_EQ(output0_shape.size(), 3);
  ASSERT_EQ(output0_shape[0], 1);
  ASSERT_EQ(output0_shape[1], 128);
  ASSERT_EQ(output0_shape[2], 768);
  const std::vector<int64_t> &output1_shape = output1->GetShape();
  ASSERT_EQ(output1_shape.size(), 3);
  ASSERT_EQ(output1_shape[0], 1);
  ASSERT_EQ(output1_shape[1], 128);
  ASSERT_EQ(output1_shape[2], 768);
  const std::vector<int64_t> &output2_shape = output2->GetShape();
  ASSERT_EQ(output2_shape.size(), 3);
  ASSERT_EQ(output2_shape[0], 1);
  ASSERT_EQ(output2_shape[1], 128);
  ASSERT_EQ(output2_shape[2], 768);
}

TEST(NodeTest, PowTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_pow_PATH);
  std::shared_ptr<Node> pow = graph.GetNode("pow");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> exponent = graph.GetEdge("exponent");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(pow, nullptr);
  ASSERT_EQ(pow->GetOp(), Node::Op::Pow);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(exponent, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<InputEdge>(input));
  ASSERT_TRUE(isa<ConstantScalarEdge>(exponent));
  ASSERT_TRUE(isa<OutputEdge>(output));
  std::shared_ptr<ConstantScalarEdge> constant_exponent =
      std::dynamic_pointer_cast<ConstantScalarEdge>(exponent);
  ASSERT_NE(constant_exponent, nullptr);
  ASSERT_EQ(constant_exponent->GetType(), Type::kFloat32);
  ASSERT_EQ(constant_exponent->GetValue(), 3);
}

TEST(NodeTest, ReshapeTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_reshape_PATH);
  std::shared_ptr<Node> reshape = graph.GetNode("reshape");
  std::shared_ptr<Edge> data = graph.GetEdge("data");
  std::shared_ptr<Edge> shape = graph.GetEdge("shape");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(reshape, nullptr);
  ASSERT_EQ(reshape->GetOp(), Node::Op::Reshape);
  ASSERT_NE(data, nullptr);
  ASSERT_NE(shape, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<InputEdge>(data));
  ASSERT_TRUE(isa<ConstantTensorEdge>(shape));
  ASSERT_TRUE(isa<OutputEdge>(output));
  ASSERT_EQ(shape->GetType(), Type::kInt64);
  const std::vector<int64_t> &shape_array = shape->GetShape();
  ASSERT_EQ(shape_array.size(), 1);
  ASSERT_EQ(shape_array[0], 4);
  std::shared_ptr<ConstantTensorEdge> constant_shape =
      std::dynamic_pointer_cast<ConstantTensorEdge>(shape);
  ASSERT_NE(constant_shape, nullptr);
  const Tensor &shape_tensor = constant_shape->GetValue();
  ASSERT_EQ(shape_tensor.GetType(), Type::kInt64);
  ASSERT_EQ(shape_tensor.Get({0}), 1);
  ASSERT_EQ(shape_tensor.Get({1}), 128);
  ASSERT_EQ(shape_tensor.Get({2}), 12);
  ASSERT_EQ(shape_tensor.Get({3}), 64);
  std::vector<std::shared_ptr<Edge>> froms = graph.GetNodeFrom(*reshape);
  ASSERT_EQ(froms.size(), 2);
  std::vector<std::shared_ptr<Edge>> to = graph.GetNodeTo(*reshape);
  ASSERT_EQ(to.size(), 1);
}

TEST(NodeTest, SoftmaxTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_softmax_PATH);
  std::shared_ptr<Node> softmax = graph.GetNode("softmax");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(softmax, nullptr);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(softmax->GetOp(), Node::Op::Softmax);
  const std::vector<int64_t> &input_shape = input->GetShape();
  const std::vector<int64_t> &output_shape = output->GetShape();
  ASSERT_EQ(input_shape.size(), 4);
  ASSERT_EQ(output_shape.size(), 4);
  ASSERT_EQ(input_shape[0], 1);
  ASSERT_EQ(input_shape[1], 12);
  ASSERT_EQ(input_shape[2], 128);
  ASSERT_EQ(input_shape[3], 768);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 12);
  ASSERT_EQ(output_shape[2], 128);
  ASSERT_EQ(output_shape[3], 768);
  Attribute axis = softmax->GetAttribute("axis");
  ASSERT_EQ(axis.GetType(), Attribute::Type::Int64);
  ASSERT_EQ(axis.GetInt64(), -1);
}

TEST(NodeTest, SubTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_sub_PATH);
  std::shared_ptr<Node> sub = graph.GetNode("sub");
  std::shared_ptr<Edge> diff = graph.GetEdge("diff");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(sub, nullptr);
  ASSERT_NE(diff, nullptr);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(sub->GetOp(), Node::Op::Sub);
  const std::vector<int64_t> &diffShape = diff->GetShape();
  const std::vector<int64_t> &input_shape = input->GetShape();
  const std::vector<int64_t> &output_shape = output->GetShape();
  ASSERT_EQ(diffShape.size(), 0);
  ASSERT_EQ(input_shape.size(), 4);
  ASSERT_EQ(output_shape.size(), 4);
  ASSERT_EQ(input_shape[0], 1);
  ASSERT_EQ(input_shape[1], 1);
  ASSERT_EQ(input_shape[2], 1);
  ASSERT_EQ(input_shape[3], 128);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 1);
  ASSERT_EQ(output_shape[2], 1);
  ASSERT_EQ(output_shape[3], 128);
}

TEST(NodeTest, TanhTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_tanh_PATH);
  std::shared_ptr<Node> tanh = graph.GetNode("tanh");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(tanh, nullptr);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(tanh->GetOp(), Node::Op::Tanh);
  const std::vector<int64_t> &input_shape = input->GetShape();
  const std::vector<int64_t> &output_shape = output->GetShape();
  ASSERT_EQ(input_shape.size(), 2);
  ASSERT_EQ(output_shape.size(), 2);
  ASSERT_EQ(input_shape[0], 1);
  ASSERT_EQ(input_shape[1], 768);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 768);
}

TEST(NodeTest, TransposeTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_transpose_PATH);
  std::shared_ptr<Node> transpose = graph.GetNode("transpose");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(transpose, nullptr);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(transpose->GetOp(), Node::Op::Transpose);
  const std::vector<int64_t> &input_shape = input->GetShape();
  const std::vector<int64_t> &output_shape = output->GetShape();
  ASSERT_EQ(input_shape.size(), 4);
  ASSERT_EQ(output_shape.size(), 4);
  ASSERT_EQ(input_shape[0], 1);
  ASSERT_EQ(input_shape[1], 128);
  ASSERT_EQ(input_shape[2], 12);
  ASSERT_EQ(input_shape[3], 64);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 12);
  ASSERT_EQ(output_shape[2], 128);
  ASSERT_EQ(output_shape[3], 64);
  Attribute perm = transpose->GetAttribute("perm");
  ASSERT_EQ(perm.GetType(), Attribute::Type::Int64Array);
  const std::vector<int64_t> &perm_array = perm.GetInt64Array();
  ASSERT_EQ(perm_array.size(), 4);
  ASSERT_EQ(perm_array[0], 0);
  ASSERT_EQ(perm_array[1], 2);
  ASSERT_EQ(perm_array[2], 1);
  ASSERT_EQ(perm_array[3], 3);
}

TEST(NodeTest, UnsqueezeTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_unsqueeze_PATH);
  std::shared_ptr<Node> unsqueeze = graph.GetNode("unsqueeze");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(unsqueeze, nullptr);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(unsqueeze->GetOp(), Node::Op::Unsqueeze);
  const std::vector<int64_t> &input_shape = input->GetShape();
  const std::vector<int64_t> &output_shape = output->GetShape();
  ASSERT_EQ(input_shape.size(), 2);
  ASSERT_EQ(output_shape.size(), 4);
  ASSERT_EQ(input_shape[0], 1);
  ASSERT_EQ(input_shape[1], 128);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 1);
  ASSERT_EQ(output_shape[2], 1);
  ASSERT_EQ(output_shape[3], 128);
}

TEST(NodeTest, WhereTest) {
  std::unique_ptr<Parser> parser = Parser::Make();
  Graph graph = parser->Run(ONNX_where_PATH);
  std::shared_ptr<Node> where = graph.GetNode("where");
  std::shared_ptr<Edge> condition = graph.GetEdge("condition");
  std::shared_ptr<Edge> input = graph.GetEdge("input");
  std::shared_ptr<Edge> other = graph.GetEdge("other");
  std::shared_ptr<Edge> output = graph.GetEdge("output");
  ASSERT_NE(where, nullptr);
  ASSERT_EQ(where->GetOp(), Node::Op::Where);
  ASSERT_NE(condition, nullptr);
  ASSERT_NE(input, nullptr);
  ASSERT_NE(other, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(isa<ConstantEdge>(condition));
  ASSERT_TRUE(isa<InputEdge>(input));
  ASSERT_TRUE(isa<ConstantEdge>(other));
  ASSERT_TRUE(isa<OutputEdge>(output));
}
