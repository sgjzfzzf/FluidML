#include "worker/scheduler.h"
#include "structure/flow/node.h"
#include "structure/kernel/add.h"
#include "structure/kernel/div.h"
#include "structure/kernel/erf.h"
#include "structure/kernel/gather.h"
#include "structure/kernel/gemm.h"
#include "structure/kernel/layernormalization.h"
#include "structure/kernel/matmul.h"
#include "structure/kernel/mul.h"
#include "structure/kernel/pow.h"
#include "structure/kernel/reshape.h"
#include "structure/kernel/softmax.h"
#include "structure/kernel/split.h"
#include "structure/kernel/sub.h"
#include "structure/kernel/tanh.h"
#include "structure/kernel/transpose.h"
#include "structure/kernel/unsqueeze.h"
#include "structure/kernel/where.h"
#include "structure/tensor/tensor.h"
#include <memory>
#include <string>
#include <unordered_map>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace worker {
void NaiveScheduler::Run(
    mlir::OpBuilder &builder, const flow::Sequence &sequence,
    std::unordered_map<std::string, mlir::Value> &symbol_table) {
  const std::vector<std::shared_ptr<flow::Node>> &nodes = sequence.GetNodes();
  for (std::shared_ptr<flow::Node> node : nodes) {
    if (std::shared_ptr<flow::AddConstantScalarNode> ptr =
            std::dynamic_pointer_cast<flow::AddConstantScalarNode>(node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      Type type = ptr->GetType();
      float64_t constant = ptr->GetValue();
      kernel::AddConstantScalarKernel kernel(type, constant);
      kernel.Run(builder, input, output);
    } else if (std::shared_ptr<flow::AddConstantTensorNode> ptr =
                   std::dynamic_pointer_cast<flow::AddConstantTensorNode>(
                       node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      Tensor tensor = ptr->GetTensor();
      kernel::AddConstTensorKernel kernel(std::move(tensor));
      kernel.Run(builder, input, output);
    } else if (std::shared_ptr<flow::AddCommonNode> ptr =
                   std::dynamic_pointer_cast<flow::AddCommonNode>(node)) {
      std::shared_ptr<flow::Edge> lhs_edge = ptr->GetLhs();
      std::shared_ptr<flow::Edge> rhs_edge = ptr->GetRhs();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &lhs_name = lhs_edge->GetName();
      const std::string &rhs_name = rhs_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &lhs = symbol_table.at(lhs_name);
      mlir::Value &rhs = symbol_table.at(rhs_name);
      mlir::Value &output = symbol_table.at(output_name);
      kernel::AddCommonKernel kernel;
      kernel.Run(builder, lhs, rhs, output);
    } else if (std::shared_ptr<flow::DivConstantScalarNode> ptr =
                   std::dynamic_pointer_cast<flow::DivConstantScalarNode>(
                       node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      Type type = ptr->GetType();
      float64_t constant = ptr->GetValue();
      kernel::DivConstScalarKernel kernel(type, constant);
      kernel.Run(builder, input, output);
    } else if (std::shared_ptr<flow::ErfNode> ptr =
                   std::dynamic_pointer_cast<flow::ErfNode>(node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      kernel::ErfKernel kernel;
      kernel.Run(builder, input, output);
    } else if (std::shared_ptr<flow::GatherConstantIndexScalarNode> ptr =
                   std::dynamic_pointer_cast<
                       flow::GatherConstantIndexScalarNode>(node)) {
      std::shared_ptr<flow::Edge> lhs_edge = ptr->GetLhs();
      int64_t rhs = ptr->GetRhs();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      int64_t axis = ptr->GetAxis();
      const std::string &lhs_name = lhs_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &lhs = symbol_table.at(lhs_name);
      mlir::Value &output = symbol_table.at(output_name);
      kernel::GatherConstantIndexScalarKernel kernel(axis);
      kernel.Run(builder, lhs, rhs, output);
    } else if (std::shared_ptr<flow::GatherConstantDataTensorNode> ptr =
                   std::dynamic_pointer_cast<
                       flow::GatherConstantDataTensorNode>(node)) {
      const Tensor &lhs = ptr->GetLhs();
      std::shared_ptr<flow::Edge> rhs_edge = ptr->GetRhs();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &rhs_name = rhs_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &rhs = symbol_table.at(rhs_name);
      mlir::Value &output = symbol_table.at(output_name);
      kernel::GatherConstantDataTensorKernel kernel;
      kernel.Run(builder, lhs, rhs, output);
    } else if (std::shared_ptr<flow::GemmConstantWeightsBiasNode> ptr =
                   std::dynamic_pointer_cast<flow::GemmConstantWeightsBiasNode>(
                       node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      const Tensor &weights = ptr->GetWeights();
      const Tensor &bias = ptr->GetBias();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      float64_t alpha = ptr->GetAlpha();
      float64_t beta = ptr->GetBeta();
      bool transA = ptr->GetTransA();
      bool transB = ptr->GetTransB();
      kernel::GemmConstantWeightsBiasKernel kernel(alpha, beta, transA, transB);
      kernel.Run(builder, input, weights, bias, output);
    } else if (std::shared_ptr<flow::LayerNormalizationConstantScaleBiasNode>
                   ptr = std::dynamic_pointer_cast<
                       flow::LayerNormalizationConstantScaleBiasNode>(node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      const Tensor &scale = ptr->GetScale();
      const Tensor &bias = ptr->GetBias();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &name = ptr->GetName();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      mlir::Value &buffer = symbol_table.at(name);
      int64_t axis = ptr->GetAxis();
      float64_t epsilon = ptr->GetEpsilon();
      kernel::LayerNormalizationConstantScaleBiasKernel kernel(axis, epsilon);
      kernel.Run(builder, input, scale, bias, output, buffer);
    } else if (std::shared_ptr<flow::MatMulConstantLhsNode> ptr =
                   std::dynamic_pointer_cast<flow::MatMulConstantLhsNode>(
                       node)) {
      const Tensor &lhs = ptr->GetLhs();
      std::shared_ptr<flow::Edge> rhs_edge = ptr->GetRhs();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      std::string rhs_name = rhs_edge->GetName();
      std::string output_name = output_edge->GetName();
      mlir::Value &rhs = symbol_table.at(rhs_name);
      mlir::Value &output = symbol_table.at(output_name);
      kernel::MatMulConstantLhsKernel kernel;
      kernel.Run(builder, lhs, rhs, output);
    } else if (std::shared_ptr<flow::MatMulConstantRhsNode> ptr =
                   std::dynamic_pointer_cast<flow::MatMulConstantRhsNode>(
                       node)) {
      std::shared_ptr<flow::Edge> lhs = ptr->GetLhs();
      const Tensor &rhs = ptr->GetRhs();
      std::shared_ptr<flow::Edge> output = ptr->GetOutput();
      std::string lhs_name = lhs->GetName();
      std::string output_name = output->GetName();
      mlir::Value &lhs_value = symbol_table.at(lhs_name);
      mlir::Value &output_value = symbol_table.at(output_name);
      kernel::MatMulConstantRhsKernel kernel;
      kernel.Run(builder, lhs_value, rhs, output_value);
    } else if (std::shared_ptr<flow::MatMulCommonNode> ptr =
                   std::dynamic_pointer_cast<flow::MatMulCommonNode>(node)) {
      std::shared_ptr<flow::Edge> lhs = ptr->GetLhs();
      std::shared_ptr<flow::Edge> rhs = ptr->GetRhs();
      std::shared_ptr<flow::Edge> output = ptr->GetOutput();
      std::string lhs_name = lhs->GetName();
      std::string rhs_name = rhs->GetName();
      std::string output_name = output->GetName();
      mlir::Value &lhs_value = symbol_table.at(lhs_name);
      mlir::Value &rhs_value = symbol_table.at(rhs_name);
      mlir::Value &output_value = symbol_table.at(output_name);
      kernel::MatMulCommonKernel kernel;
      kernel.Run(builder, lhs_value, rhs_value, output_value);
    } else if (std::shared_ptr<flow::MulConstantScalarNode> ptr =
                   std::dynamic_pointer_cast<flow::MulConstantScalarNode>(
                       node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      Type type = ptr->GetType();
      float64_t constant = ptr->GetValue();
      kernel::MulConstantScalarKernel kernel(type, constant);
      kernel.Run(builder, input, output);
    } else if (std::shared_ptr<flow::MulConstantTensorNode> ptr =
                   std::dynamic_pointer_cast<flow::MulConstantTensorNode>(
                       node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      Tensor tensor = ptr->GetTensor();
      kernel::MulConstantTensorKernel kernel(std::move(tensor));
      kernel.Run(builder, input, output);
    } else if (std::shared_ptr<flow::MulCommonNode> ptr =
                   std::dynamic_pointer_cast<flow::MulCommonNode>(node)) {
      std::shared_ptr<flow::Edge> lhs_edge = ptr->GetLhs();
      std::shared_ptr<flow::Edge> rhs_edge = ptr->GetRhs();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &lhs_name = lhs_edge->GetName();
      const std::string &rhs_name = rhs_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &lhs = symbol_table.at(lhs_name);
      mlir::Value &rhs = symbol_table.at(rhs_name);
      mlir::Value &output = symbol_table.at(output_name);
      kernel::MulCommonKernel kernel;
      kernel.Run(builder, lhs, rhs, output);
    } else if (std::shared_ptr<flow::PowNode> ptr =
                   std::dynamic_pointer_cast<flow::PowNode>(node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      Type type = ptr->GetType();
      const float64_t exp = ptr->GetExp();
      kernel::PowKernel kernel(type, exp);
      kernel.Run(builder, input, output);
    } else if (std::shared_ptr<flow::ReshapeNode> ptr =
                   std::dynamic_pointer_cast<flow::ReshapeNode>(node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      kernel::ReshapeKernel kernel;
      kernel.Run(builder, input, output);
    } else if (std::shared_ptr<flow::SoftmaxNode> ptr =
                   std::dynamic_pointer_cast<flow::SoftmaxNode>(node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &name = ptr->GetName();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      mlir::Value &buffer = symbol_table.at(name);
      const int64_t axis = ptr->GetAxis();
#ifdef DEBUG
      assert(axis >= 0);
#endif
      kernel::SoftmaxKernel kernel(axis);
      kernel.Run(builder, input, output, buffer);
    } else if (std::shared_ptr<flow::SplitNode> ptr =
                   std::dynamic_pointer_cast<flow::SplitNode>(node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::vector<std::shared_ptr<flow::Edge>> output_edges = ptr->GetOutputs();
      const std::string &input_name = input_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      llvm::SmallVector<mlir::Value> outputs;
      for (std::shared_ptr<flow::Edge> output_edge : output_edges) {
        const std::string &output_name = output_edge->GetName();
        mlir::Value &output = symbol_table.at(output_name);
        outputs.push_back(output);
      }
      const int64_t axis = ptr->GetAxis();
#ifdef DEBUG
      assert(axis >= 0);
#endif
      kernel::SplitKernel kernel(axis);
      kernel.Run(builder, input, outputs);
    } else if (std::shared_ptr<flow::SubConstantScalarLhsNode> ptr =
                   std::dynamic_pointer_cast<flow::SubConstantScalarLhsNode>(
                       node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      Type type = ptr->GetType();
      const float64_t value = ptr->GetValue();
      kernel::SubConstantScalarLhsKernel kernel(type, value);
      kernel.Run(builder, input, output);
    } else if (std::shared_ptr<flow::TanhNode> ptr =
                   std::dynamic_pointer_cast<flow::TanhNode>(node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      kernel::TanhKernel kernel;
      kernel.Run(builder, input, output);
    } else if (std::shared_ptr<flow::TransposeNode> ptr =
                   std::dynamic_pointer_cast<flow::TransposeNode>(node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      const std::vector<int64_t> &perm = ptr->GetPerm();
      kernel::TransposeKernel kernel(perm);
      kernel.Run(builder, input, output);
    } else if (std::shared_ptr<flow::UnsqueezeNode> ptr =
                   std::dynamic_pointer_cast<flow::UnsqueezeNode>(node)) {
      std::shared_ptr<flow::Edge> input_edge = ptr->GetInput();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &input_name = input_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      std::vector<int64_t> axes = ptr->GetAxes();
      kernel::UnSqueezeKernel kernel(std::move(axes));
      kernel.Run(builder, input, output);
    } else if (std::shared_ptr<flow::WhereConstantCondConstantScalarYNode> ptr =
                   std::dynamic_pointer_cast<
                       flow::WhereConstantCondConstantScalarYNode>(node)) {
      const Tensor &cond = ptr->GetCond();
      Type type = ptr->GetType();
      float64_t y = ptr->GetY();
      std::shared_ptr<flow::Edge> x_edge = ptr->GetX();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &x_name = x_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &x = symbol_table.at(x_name);
      mlir::Value &output = symbol_table.at(output_name);
      kernel::WhereConstantCondConstantScalarYKernel kernel;
      kernel.Run(builder, cond, x, type, y, output);
    } else if (std::shared_ptr<flow::WhereConstantCondConstantTensorYNode> ptr =
                   std::dynamic_pointer_cast<
                       flow::WhereConstantCondConstantTensorYNode>(node)) {
      const Tensor &cond = ptr->GetCond();
      std::shared_ptr<flow::Edge> x_edge = ptr->GetX();
      const Tensor &y = ptr->GetY();
      std::shared_ptr<flow::Edge> output_edge = ptr->GetOutput();
      const std::string &x_name = x_edge->GetName();
      const std::string &output_name = output_edge->GetName();
      mlir::Value &x = symbol_table.at(x_name);
      mlir::Value &output = symbol_table.at(output_name);
      kernel::WhereConstantCondConstantTensorYKernel kernel;
      kernel.Run(builder, cond, x, y, output);
    }
  }
}
} // namespace worker
} // namespace cpu_transformers
