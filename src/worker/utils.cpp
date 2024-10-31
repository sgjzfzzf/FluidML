#include "worker/utils.h"
#include "fmt/core.h"
#include "structure/flow/node.h"
#include "structure/flow/region.h"
#include "structure/kernel/generator/add.h"
#include "structure/kernel/generator/add_div_erf_add_mul_mul.h"
#include "structure/kernel/generator/cast.h"
#include "structure/kernel/generator/concat.h"
#include "structure/kernel/generator/conv.h"
#include "structure/kernel/generator/cum_sum.h"
#include "structure/kernel/generator/div.h"
#include "structure/kernel/generator/dropout.h"
#include "structure/kernel/generator/equal.h"
#include "structure/kernel/generator/erf.h"
#include "structure/kernel/generator/flatten.h"
#include "structure/kernel/generator/gather.h"
#include "structure/kernel/generator/gather_add_add.h"
#include "structure/kernel/generator/gemm.h"
#include "structure/kernel/generator/layer_normalization.h"
#include "structure/kernel/generator/matmul.h"
#include "structure/kernel/generator/maxpool.h"
#include "structure/kernel/generator/mul.h"
#include "structure/kernel/generator/neg.h"
#include "structure/kernel/generator/not.h"
#include "structure/kernel/generator/pad.h"
#include "structure/kernel/generator/pow.h"
#include "structure/kernel/generator/reduce_mean.h"
#include "structure/kernel/generator/relu.h"
#include "structure/kernel/generator/reshape.h"
#include "structure/kernel/generator/slice.h"
#include "structure/kernel/generator/softmax.h"
#include "structure/kernel/generator/sqrt.h"
#include "structure/kernel/generator/sub.h"
#include "structure/kernel/generator/tanh.h"
#include "structure/kernel/generator/transpose.h"
#include "structure/kernel/generator/unsqueeze.h"
#include "structure/kernel/generator/unsqueeze_sub_mul.h"
#include "structure/kernel/generator/where.h"
#include "structure/kernel/kernel/add.h"
#include "structure/kernel/kernel/cast.h"
#include "structure/kernel/kernel/concat.h"
#include "structure/kernel/kernel/conv.h"
#include "structure/kernel/kernel/cum_sum.h"
#include "structure/kernel/kernel/div.h"
#include "structure/kernel/kernel/dropout.h"
#include "structure/kernel/kernel/equal.h"
#include "structure/kernel/kernel/flatten.h"
#include "structure/kernel/kernel/maxpool.h"
#include "structure/kernel/kernel/mul.h"
#include "structure/kernel/kernel/neg.h"
#include "structure/kernel/kernel/not.h"
#include "structure/kernel/kernel/pad.h"
#include "structure/kernel/kernel/reduce_mean.h"
#include "structure/kernel/kernel/relu.h"
#include "structure/kernel/kernel/slice.h"
#include "structure/kernel/kernel/sqrt.h"
#include "structure/kernel/kernel/sub.h"
#include "structure/kernel/kernel/transpose.h"
#include "structure/tensor/meta.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace worker {

std::string GetBufferName(std::string_view name) {
  return fmt::format("{}-buffer", name);
}

std::unique_ptr<kernel::Kernel> SelectKernel(const flow::Node *node) {
  std::unique_ptr<kernel::Kernel> kernel = nullptr;
  if (const flow::AddConstantNode *ptr =
          dynamic_cast<const flow::AddConstantNode *>(node)) {
    Type type = ptr->GetType();
    float64_t constant = ptr->GetValue();
    kernel = std::make_unique<kernel::AddConstantKernel>(type, constant);
  } else if (const flow::AddCommonNode *ptr =
                 dynamic_cast<const flow::AddCommonNode *>(node)) {
    kernel = std::make_unique<kernel::AddCommonKernel>();
  } else if (const flow::AddDivErfAddMulMulNode *ptr =
                 dynamic_cast<const flow::AddDivErfAddMulMulNode *>(node)) {
    Tensor add0_weight = ptr->GetAdd0Weight();
    Type div_type = ptr->GetDivType(), add1_type = ptr->GetAdd1Type(),
         mul1_type = ptr->GetMul1Type();
    float64_t div_weight = ptr->GetDivWeight(),
              add1_weight = ptr->GetAdd1Weight(),
              mul1_weight = ptr->GetMul1Weight();
    kernel = std::make_unique<kernel::AddDivErfAddMulMulKernel>(
        std::move(add0_weight), div_type, div_weight, add1_type, add1_weight,
        mul1_type, mul1_weight);
  } else if (const flow::CastNode *ptr =
                 dynamic_cast<const flow::CastNode *>(node)) {
    kernel = std::make_unique<kernel::CastKernel>();
  } else if (const flow::Concat2CommonNode *ptr =
                 dynamic_cast<const flow::Concat2CommonNode *>(node)) {
    const int64_t axis = ptr->GetAxis();
    kernel = std::make_unique<kernel::Concat2Kernel>(axis);
  } else if (const flow::ConvWithoutPaddingNode *ptr =
                 dynamic_cast<const flow::ConvWithoutPaddingNode *>(node)) {
    std::vector<int64_t> dilations = ptr->GetDilations(),
                         kernel_shape = ptr->GetKernelShape(),
                         strides = ptr->GetStrides();
    const int64_t group = ptr->GetGroup();
    std::optional<Tensor> bias = ptr->GetBias();
    kernel = std::make_unique<kernel::ConvWithoutPaddingKernel>(
        std::move(dilations), group, std::move(kernel_shape),
        std::move(strides), std::move(bias));
  } else if (const flow::ConvWithPaddingNode *ptr =
                 dynamic_cast<const flow::ConvWithPaddingNode *>(node)) {
    std::vector<int64_t> dilations = ptr->GetDilations(),
                         kernel_shape = ptr->GetKernelShape(),
                         pads = ptr->GetPads(), strides = ptr->GetStrides();
    const int64_t group = ptr->GetGroup();
    std::optional<Tensor> bias = ptr->GetBias();
    kernel = std::make_unique<kernel::ConvWithPaddingKernel>(
        std::move(dilations), group, std::move(kernel_shape), std::move(pads),
        std::move(strides), std::move(bias));
  } else if (const flow::CumSumNode *ptr =
                 dynamic_cast<const flow::CumSumNode *>(node)) {
    const int64_t axis = ptr->GetAxis();
    bool exclusive = ptr->GetExclusive();
    bool reverse = ptr->GetReverse();
    kernel = std::make_unique<kernel::CumSumKernel>(axis, exclusive, reverse);
  } else if (const flow::DivConstantRhsNode *ptr =
                 dynamic_cast<const flow::DivConstantRhsNode *>(node)) {
    Type type = ptr->GetType();
    float64_t constant = ptr->GetValue();
    kernel = std::make_unique<kernel::DivConstantRhsKernel>(type, constant);
  } else if (const flow::DivCommonNode *ptr =
                 dynamic_cast<const flow::DivCommonNode *>(node)) {
    kernel = std::make_unique<kernel::DivCommonKernel>();
  } else if (const flow::DropoutNode *ptr =
                 dynamic_cast<const flow::DropoutNode *>(node)) {
    float64_t ratio = ptr->GetRatio();
    kernel = std::make_unique<kernel::DropoutKernel>(ratio);
  } else if (const flow::EqualNode *ptr =
                 dynamic_cast<const flow::EqualNode *>(node)) {
    Type type = ptr->GetType();
    float64_t value = ptr->GetValue();
    kernel = std::make_unique<kernel::EqualKernel>(type, value);
  } else if (const flow::ErfNode *ptr =
                 dynamic_cast<const flow::ErfNode *>(node)) {
    kernel = std::make_unique<kernel::ErfKernel>();
  } else if (const flow::FlattenNode *ptr =
                 dynamic_cast<const flow::FlattenNode *>(node)) {
    const int64_t axis = ptr->GetAxis();
    kernel = std::make_unique<kernel::FlattenKernel>(axis);
  } else if (const flow::GatherConstantIndexScalarNode *ptr =
                 dynamic_cast<const flow::GatherConstantIndexScalarNode *>(
                     node)) {
    const int64_t axis = ptr->GetAxis(), index = ptr->GetIndex();
    kernel =
        std::make_unique<kernel::GatherConstantIndexScalarKernel>(axis, index);
  } else if (const flow::GatherConstantIndicesTensorNode *ptr =
                 dynamic_cast<const flow::GatherConstantIndicesTensorNode *>(
                     node)) {
    Tensor indices = ptr->GetIndices();
    const int64_t axis = ptr->GetAxis();
    kernel = std::make_unique<kernel::GatherConstantIndicesTensorKernel>(
        std::move(indices), axis);
  } else if (const flow::GatherConstantDataTensorNode *ptr =
                 dynamic_cast<const flow::GatherConstantDataTensorNode *>(
                     node)) {
    Tensor data = ptr->GetData();
    kernel = std::make_unique<kernel::GatherConstantDataTensorKernel>(
        std::move(data));
  } else if (const flow::GatherConstantDataTensorAddTensorLhsAddTensorLhsNode
                 *ptr = dynamic_cast<
                     const flow::
                         GatherConstantDataTensorAddTensorLhsAddTensorLhsNode
                             *>(node)) {
    Tensor data = ptr->GetData(), add0_weight = ptr->GetAdd0Weight(),
           add1_weight = ptr->GetAdd1Weight();
    kernel = std::make_unique<
        kernel::GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel>(
        std::move(data), std::move(add0_weight), std::move(add1_weight));
  } else if (const flow::GemmConstantWeightsBiasNode *ptr =
                 dynamic_cast<const flow::GemmConstantWeightsBiasNode *>(
                     node)) {
    Tensor bias = ptr->GetBias();
    float64_t alpha = ptr->GetAlpha(), beta = ptr->GetBeta();
    bool transA = ptr->GetTransA(), transB = ptr->GetTransB();
    kernel = std::make_unique<kernel::GemmConstantBiasKernel>(
        alpha, beta, transA, transB, std::move(bias));
  } else if (const flow::LayerNormalizationConstantScaleBiasNode *ptr =
                 dynamic_cast<
                     const flow::LayerNormalizationConstantScaleBiasNode *>(
                     node)) {
    Tensor scale = ptr->GetScale(), bias = ptr->GetBias();
    const int64_t axis = ptr->GetAxis();
    float64_t epsilon = ptr->GetEpsilon();
    kernel =
        std::make_unique<kernel::LayerNormalizationConstantScaleBiasKernel>(
            axis, epsilon, std::move(scale), std::move(bias));
  } else if (const flow::MatMulNode *ptr =
                 dynamic_cast<const flow::MatMulNode *>(node)) {
    kernel = std::make_unique<kernel::MatMulKernel>();
  } else if (const flow::MaxPoolWithoutPaddingNode *ptr =
                 dynamic_cast<const flow::MaxPoolWithoutPaddingNode *>(node)) {
    std::vector<int64_t> kernel_shape = ptr->GetKernelShape(),
                         strides = ptr->GetStrides();
    kernel = std::make_unique<kernel::MaxPoolWithoutPaddingKernel>(
        std::move(kernel_shape), std::move(strides));
  } else if (const flow::MulConstantNode *ptr =
                 dynamic_cast<const flow::MulConstantNode *>(node)) {
    Type type = ptr->GetType();
    float64_t constant = ptr->GetValue();
    kernel = std::make_unique<kernel::MulConstantKernel>(type, constant);
  } else if (const flow::MulCommonNode *ptr =
                 dynamic_cast<const flow::MulCommonNode *>(node)) {
    kernel = std::make_unique<kernel::MulCommonKernel>();
  } else if (const flow::NegNode *ptr =
                 dynamic_cast<const flow::NegNode *>(node)) {
    kernel = std::make_unique<kernel::NegKernel>();
  } else if (const flow::NotNode *ptr =
                 dynamic_cast<const flow::NotNode *>(node)) {
    kernel = std::make_unique<kernel::NotKernel>();
  } else if (const flow::PadNode *ptr =
                 dynamic_cast<const flow::PadNode *>(node)) {
    std::vector<std::tuple<int64_t, int64_t>> pads = ptr->GetPads();
    kernel = std::make_unique<kernel::PadKernel>(std::move(pads));
  } else if (const flow::PowNode *ptr =
                 dynamic_cast<const flow::PowNode *>(node)) {
    Type type = ptr->GetType();
    const float64_t exp = ptr->GetExp();
    kernel = std::make_unique<kernel::PowKernel>(type, exp);
  } else if (const flow::ReduceMeanNode *ptr =
                 dynamic_cast<const flow::ReduceMeanNode *>(node)) {
    const std::vector<int64_t> &axes_ref = ptr->GetAxes();
    llvm::SmallVector<int64_t> axes(axes_ref.begin(), axes_ref.end());
    const bool keep_dims = ptr->GetKeepDims();
    kernel =
        std::make_unique<kernel::ReduceMeanKernel>(std::move(axes), keep_dims);
  } else if (const flow::ReluNode *ptr =
                 dynamic_cast<const flow::ReluNode *>(node)) {
    kernel = std::make_unique<kernel::ReluKernel>();
  } else if (const flow::ReshapeNode *ptr =
                 dynamic_cast<const flow::ReshapeNode *>(node)) {
    kernel = std::make_unique<kernel::ReshapeKernel>();
  } else if (const flow::SliceNode *ptr =
                 dynamic_cast<const flow::SliceNode *>(node)) {
    const std::vector<int64_t> &starts = ptr->GetStarts(),
                               ends = ptr->GetEnds(), axes = ptr->GetAxes(),
                               steps = ptr->GetSteps();
    const size_t len = starts.size();
#ifdef DEBUG
    assert(len == ends.size());
    assert(len == axes.size());
    assert(len == steps.size());
#endif
    llvm::SmallVector<llvm::SmallVector<int64_t, 4>> informations;
    for (size_t i = 0; i < len; ++i) {
      informations.push_back({starts[i], ends[i], axes[i], steps[i]});
    }
    kernel = std::make_unique<kernel::SliceKernel>(std::move(informations));
  } else if (const flow::SoftmaxNode *ptr =
                 dynamic_cast<const flow::SoftmaxNode *>(node)) {
    const int64_t axis = ptr->GetAxis();
#ifdef DEBUG
    assert(axis >= 0);
#endif
    kernel = std::make_unique<kernel::SoftmaxKernel>(axis);
  } else if (const flow::SqrtNode *ptr =
                 dynamic_cast<const flow::SqrtNode *>(node)) {
    kernel = std::make_unique<kernel::SqrtKernel>();
  } else if (const flow::SubConstantLhsNode *ptr =
                 dynamic_cast<const flow::SubConstantLhsNode *>(node)) {
    Type type = ptr->GetType();
    const float64_t value = ptr->GetValue();
    kernel = std::make_unique<kernel::SubConstantLhsKernel>(type, value);
  } else if (const flow::SubCommonNode *ptr =
                 dynamic_cast<const flow::SubCommonNode *>(node)) {
    kernel = std::make_unique<kernel::SubCommonKernel>();
  } else if (const flow::TanhNode *ptr =
                 dynamic_cast<const flow::TanhNode *>(node)) {
    kernel = std::make_unique<kernel::TanhKernel>();
  } else if (const flow::TransposeNode *ptr =
                 dynamic_cast<const flow::TransposeNode *>(node)) {
    std::vector<int64_t> perm = ptr->GetPerm();
    kernel = std::make_unique<kernel::TransposeKernel>(std::move(perm));
  } else if (const flow::UnsqueezeNode *ptr =
                 dynamic_cast<const flow::UnsqueezeNode *>(node)) {
    std::vector<int64_t> axes = ptr->GetAxes();
    kernel = std::make_unique<kernel::UnsqueezeKernel>(std::move(axes));
  } else if (const flow::UnsqueezeSubLhsScalarMulRhsScalarNode *ptr =
                 dynamic_cast<const flow::UnsqueezeSubLhsScalarMulRhsScalarNode
                                  *>(node)) {
    std::vector<int64_t> unsqueeze_axes = ptr->GetUnsqueezeAxes();
    Type sub_type = ptr->GetSubType(), mul_type = ptr->GetMulType();
    float64_t sub_val = ptr->GetSubVal(), mul_val = ptr->GetMulVal();
    kernel = std::make_unique<kernel::UnsqueezeSubLhsScalarMulRhsScalarKernel>(
        std::move(unsqueeze_axes), sub_type, sub_val, mul_type, mul_val);
  } else if (const flow::WhereConstantCondConstantScalarYNode *ptr =
                 dynamic_cast<const flow::WhereConstantCondConstantScalarYNode
                                  *>(node)) {
    Tensor cond = ptr->GetCond();
    Type type = ptr->GetType();
    float64_t y = ptr->GetY();
    kernel = std::make_unique<kernel::WhereConstantCondConstantScalarYKernel>(
        std::move(cond), type, y);
  } else if (const flow::WhereConstantCondConstantTensorYNode *ptr =
                 dynamic_cast<const flow::WhereConstantCondConstantTensorYNode
                                  *>(node)) {
    Tensor cond = ptr->GetCond(), y = ptr->GetY();
    kernel = std::make_unique<kernel::WhereConstantCondConstantTensorYKernel>(
        std::move(cond), std::move(y));
  }
  return kernel;
}

std::unique_ptr<kernel::KernelGenerator>
SelectKernelGenerator(const flow::Node *node) {
  std::unique_ptr<kernel::KernelGenerator> generator = nullptr;
  if (const flow::AddConstantNode *ptr =
          dynamic_cast<const flow::AddConstantNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    Type type = ptr->GetType();
    float64_t constant = ptr->GetValue();
    generator = kernel::AddConstantKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), type, constant);
  } else if (const flow::AddCommonNode *ptr =
                 dynamic_cast<const flow::AddCommonNode *>(node)) {
    std::shared_ptr<flow::Region> lhs = ptr->GetLhs(), rhs = ptr->GetRhs(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(output != nullptr);
#endif
    Meta lhs_meta = lhs->GetMeta(), rhs_meta = rhs->GetMeta(),
         output_meta = output->GetMeta();
    generator = kernel::AddCommonKernelGenerator::Make(
        std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta));
  } else if (const flow::AddDivErfAddMulMulNode *ptr =
                 dynamic_cast<const flow::AddDivErfAddMulMulNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    Tensor add0_weight = ptr->GetAdd0Weight();
    Type div_type = ptr->GetDivType(), add1_type = ptr->GetAdd1Type(),
         mul1_type = ptr->GetMul1Type();
    float64_t div_weight = ptr->GetDivWeight(),
              add1_weight = ptr->GetAdd1Weight(),
              mul1_weight = ptr->GetMul1Weight();
    generator = kernel::AddDivErfAddMulMulKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), std::move(add0_weight),
        div_type, div_weight, add1_type, add1_weight, mul1_type, mul1_weight);
  } else if (const flow::CastNode *ptr =
                 dynamic_cast<const flow::CastNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    generator = kernel::CastKernelGenerator::Make(std::move(input_meta),
                                                  std::move(output_meta));

  } else if (const flow::Concat2CommonNode *ptr =
                 dynamic_cast<const flow::Concat2CommonNode *>(node)) {
    std::shared_ptr<flow::Region> lhs = ptr->GetLhs(), rhs = ptr->GetRhs(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(output != nullptr);
#endif
    Meta lhs_meta = lhs->GetMeta(), rhs_meta = rhs->GetMeta(),
         output_meta = output->GetMeta();
    const int64_t axis = ptr->GetAxis();
    generator = kernel::Concat2KernelGenerator::Make(
        std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta), axis);
  } else if (const flow::CumSumNode *ptr =
                 dynamic_cast<const flow::CumSumNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    const int64_t axis = ptr->GetAxis();
    bool exclusive = ptr->GetExclusive();
    bool reverse = ptr->GetReverse();
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    generator = kernel::CumSumKernelGenerator::Make(std::move(input_meta),
                                                    std::move(output_meta),
                                                    axis, exclusive, reverse);
  } else if (const flow::ConvWithoutPaddingNode *ptr =
                 dynamic_cast<const flow::ConvWithoutPaddingNode *>(node)) {
    std::shared_ptr<flow::Region> lhs = ptr->GetLhs(), rhs = ptr->GetRhs(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(output != nullptr);
#endif
    Meta lhs_meta = lhs->GetMeta(), rhs_meta = rhs->GetMeta(),
         output_meta = output->GetMeta();
    std::vector<int64_t> dilations = ptr->GetDilations(),
                         kernel_shape = ptr->GetKernelShape(),
                         strides = ptr->GetStrides();
    const int64_t group = ptr->GetGroup();
    std::optional<Tensor> bias = ptr->GetBias();
    generator = kernel::ConvWithoutPaddingKernelGenerator::Make(
        std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta),
        std::move(dilations), group, std::move(kernel_shape),
        std::move(strides), std::move(bias));
  } else if (const flow::ConvWithPaddingNode *ptr =
                 dynamic_cast<const flow::ConvWithPaddingNode *>(node)) {
    std::shared_ptr<flow::Region> lhs = ptr->GetLhs(), rhs = ptr->GetRhs(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(output != nullptr);
#endif
    Meta lhs_meta = lhs->GetMeta(), rhs_meta = rhs->GetMeta(),
         output_meta = output->GetMeta();
    std::vector<int64_t> dilations = ptr->GetDilations(),
                         kernel_shape = ptr->GetKernelShape(),
                         pads = ptr->GetPads(), strides = ptr->GetStrides();
    const int64_t group = ptr->GetGroup();
    std::optional<Tensor> bias = ptr->GetBias();
    generator = kernel::ConvWithPaddingKernelGenerator::Make(
        std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta),
        std::move(dilations), group, std::move(kernel_shape), std::move(pads),
        std::move(strides), std::move(bias));
  } else if (const flow::DivConstantRhsNode *ptr =
                 dynamic_cast<const flow::DivConstantRhsNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    Type type = ptr->GetType();
    float64_t constant = ptr->GetValue();
    generator = kernel::DivConstantRhsKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), type, constant);
  } else if (const flow::DivCommonNode *ptr =
                 dynamic_cast<const flow::DivCommonNode *>(node)) {
    std::shared_ptr<flow::Region> lhs = ptr->GetLhs(), rhs = ptr->GetRhs(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(output != nullptr);
#endif
    Meta lhs_meta = lhs->GetMeta(), rhs_meta = rhs->GetMeta(),
         output_meta = output->GetMeta();
    generator = kernel::DivCommonKernelGenerator::Make(
        std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta));
  } else if (const flow::DropoutNode *ptr =
                 dynamic_cast<const flow::DropoutNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    float64_t ratio = ptr->GetRatio();
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    generator = kernel::DropoutKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), ratio);
  } else if (const flow::EqualNode *ptr =
                 dynamic_cast<const flow::EqualNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Type type = ptr->GetType();
    float64_t value = ptr->GetValue();
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    generator = kernel::EqualKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), type, value);
  } else if (const flow::ErfNode *ptr =
                 dynamic_cast<const flow::ErfNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    generator = kernel::ErfKernelGenerator::Make(std::move(input_meta),
                                                 std::move(output_meta));
  } else if (const flow::FlattenNode *ptr =
                 dynamic_cast<const flow::FlattenNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    const int64_t axis = ptr->GetAxis();
    generator = kernel::FlattenKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), axis);
  } else if (const flow::GatherConstantIndexScalarNode *ptr =
                 dynamic_cast<const flow::GatherConstantIndexScalarNode *>(
                     node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    const int64_t axis = ptr->GetAxis(), index = ptr->GetIndex();
    generator = kernel::GatherConstantIndexScalarKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), axis, index);
  } else if (const flow::GatherConstantIndicesTensorNode *ptr =
                 dynamic_cast<const flow::GatherConstantIndicesTensorNode *>(
                     node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    Tensor indices = ptr->GetIndices();
    const int64_t axis = ptr->GetAxis();
    generator = kernel::GatherConstantIndicesTensorKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), std::move(indices),
        axis);
  } else if (const flow::GatherConstantDataTensorNode *ptr =
                 dynamic_cast<const flow::GatherConstantDataTensorNode *>(
                     node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    Tensor data = ptr->GetData();
    generator = kernel::GatherConstantDataTensorKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), std::move(data));
  } else if (const flow::GatherConstantDataTensorAddTensorLhsAddTensorLhsNode
                 *ptr = dynamic_cast<
                     const flow::
                         GatherConstantDataTensorAddTensorLhsAddTensorLhsNode
                             *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    Tensor data = ptr->GetData(), add0_weight = ptr->GetAdd0Weight(),
           add1_weight = ptr->GetAdd1Weight();
    generator = kernel::
        GatherConstantDataTensorAddTensorLhsAddTensorLhsKernelGenerator::Make(
            std::move(input_meta), std::move(output_meta), std::move(data),
            std::move(add0_weight), std::move(add1_weight));
  } else if (const flow::GemmConstantWeightsBiasNode *ptr =
                 dynamic_cast<const flow::GemmConstantWeightsBiasNode *>(
                     node)) {
    std::shared_ptr<flow::Region> lhs = ptr->GetLhs(), rhs = ptr->GetRhs(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(output != nullptr);
#endif
    Meta lhs_meta = lhs->GetMeta(), rhs_meta = rhs->GetMeta(),
         output_meta = output->GetMeta();
    Tensor bias = ptr->GetBias();
    float64_t alpha = ptr->GetAlpha(), beta = ptr->GetBeta();
    bool transA = ptr->GetTransA(), transB = ptr->GetTransB();
    generator = kernel::GemmConstantBiasKernelGenerator::Make(
        std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta), alpha,
        beta, transA, transB, std::move(bias));
  } else if (const flow::LayerNormalizationConstantScaleBiasNode *ptr =
                 dynamic_cast<
                     const flow::LayerNormalizationConstantScaleBiasNode *>(
                     node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    Tensor scale = ptr->GetScale(), bias = ptr->GetBias();
    const int64_t axis = ptr->GetAxis();
    float64_t epsilon = ptr->GetEpsilon();
    generator =
        kernel::LayerNormalizationConstantScaleBiasKernelGenerator::Make(
            std::move(input_meta), std::move(output_meta), axis, epsilon,
            std::move(scale), std::move(bias));
  } else if (const flow::MatMulNode *ptr =
                 dynamic_cast<const flow::MatMulNode *>(node)) {
    std::shared_ptr<flow::Region> lhs = ptr->GetLhs(), rhs = ptr->GetRhs(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(output != nullptr);
#endif
    Meta lhs_meta = lhs->GetMeta(), rhs_meta = rhs->GetMeta(),
         output_meta = output->GetMeta();
    generator = kernel::MatMulKernelGenerator::Make(
        std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta));
  } else if (const flow::MaxPoolWithoutPaddingNode *ptr =
                 dynamic_cast<const flow::MaxPoolWithoutPaddingNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    std::vector<int64_t> kernel_shape = ptr->GetKernelShape(),
                         strides = ptr->GetStrides();
    generator = kernel::MaxPoolWithoutPaddingKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), std::move(kernel_shape),
        std::move(strides));
  } else if (const flow::MulConstantNode *ptr =
                 dynamic_cast<const flow::MulConstantNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    Type type = ptr->GetType();
    float64_t constant = ptr->GetValue();
    generator = kernel::MulConstantKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), type, constant);
  } else if (const flow::MulCommonNode *ptr =
                 dynamic_cast<const flow::MulCommonNode *>(node)) {
    std::shared_ptr<flow::Region> lhs = ptr->GetLhs(), rhs = ptr->GetRhs(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(output != nullptr);
#endif
    Meta lhs_meta = lhs->GetMeta(), rhs_meta = rhs->GetMeta(),
         output_meta = output->GetMeta();
    generator = kernel::MulCommonKernelGenerator::Make(
        std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta));
  } else if (const flow::NegNode *ptr =
                 dynamic_cast<const flow::NegNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    generator = kernel::NegKernelGenerator::Make(std::move(input_meta),
                                                 std::move(output_meta));
  } else if (const flow::NotNode *ptr =
                 dynamic_cast<const flow::NotNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    generator = kernel::NotKernelGenerator::Make(std::move(input_meta),
                                                 std::move(output_meta));
  } else if (const flow::PadNode *ptr =
                 dynamic_cast<const flow::PadNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    std::vector<std::tuple<int64_t, int64_t>> pads = ptr->GetPads();
    generator = kernel::PadKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), std::move(pads));
  } else if (const flow::PowNode *ptr =
                 dynamic_cast<const flow::PowNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    Type type = ptr->GetType();
    const float64_t exp = ptr->GetExp();
    generator = kernel::PowKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), type, exp);
  } else if (const flow::ReduceMeanNode *ptr =
                 dynamic_cast<const flow::ReduceMeanNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    const std::vector<int64_t> &axes_ref = ptr->GetAxes();
    llvm::SmallVector<int64_t> axes(axes_ref.begin(), axes_ref.end());
    const bool keep_dims = ptr->GetKeepDims();
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    generator = kernel::ReduceMeanKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), std::move(axes),
        keep_dims);
  } else if (const flow::ReluNode *ptr =
                 dynamic_cast<const flow::ReluNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    generator = kernel::ReluKernelGenerator::Make(std::move(input_meta),
                                                  std::move(output_meta));
  } else if (const flow::ReshapeNode *ptr =
                 dynamic_cast<const flow::ReshapeNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    generator = kernel::ReshapeKernelGenerator::Make(std::move(input_meta),
                                                     std::move(output_meta));
  } else if (const flow::SliceNode *ptr =
                 dynamic_cast<const flow::SliceNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    const std::vector<int64_t> &starts = ptr->GetStarts(),
                               ends = ptr->GetEnds(), axes = ptr->GetAxes(),
                               steps = ptr->GetSteps();
    const size_t len = starts.size();
#ifdef DEBUG
    assert(len == ends.size());
    assert(len == axes.size());
    assert(len == steps.size());
#endif
    llvm::SmallVector<llvm::SmallVector<int64_t, 4>> informations;
    for (size_t i = 0; i < len; ++i) {
      informations.push_back({starts[i], ends[i], axes[i], steps[i]});
    }
    generator = kernel::SliceKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), std::move(informations));
  } else if (const flow::SoftmaxNode *ptr =
                 dynamic_cast<const flow::SoftmaxNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    const int64_t axis = ptr->GetAxis();
#ifdef DEBUG
    assert(axis >= 0);
#endif
    generator = kernel::SoftmaxKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), axis);
  } else if (const flow::SqrtNode *ptr =
                 dynamic_cast<const flow::SqrtNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    generator = kernel::SqrtKernelGenerator::Make(std::move(input_meta),
                                                  std::move(output_meta));
  } else if (const flow::SubConstantLhsNode *ptr =
                 dynamic_cast<const flow::SubConstantLhsNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    Type type = ptr->GetType();
    const float64_t value = ptr->GetValue();
    generator = kernel::SubConstantLhsKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), type, value);
  } else if (const flow::SubCommonNode *ptr =
                 dynamic_cast<const flow::SubCommonNode *>(node)) {
    std::shared_ptr<flow::Region> lhs = ptr->GetLhs(), rhs = ptr->GetRhs(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(lhs != nullptr);
    assert(rhs != nullptr);
    assert(output != nullptr);
#endif
    Meta lhs_meta = lhs->GetMeta(), rhs_meta = rhs->GetMeta(),
         output_meta = output->GetMeta();
    generator = kernel::SubCommonKernelGenerator::Make(
        std::move(lhs_meta), std::move(rhs_meta), std::move(output_meta));
  } else if (const flow::TanhNode *ptr =
                 dynamic_cast<const flow::TanhNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    generator = kernel::TanhKernelGenerator::Make(std::move(input_meta),
                                                  std::move(output_meta));
  } else if (const flow::TransposeNode *ptr =
                 dynamic_cast<const flow::TransposeNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    std::vector<int64_t> perm = ptr->GetPerm();
    generator = kernel::TransposeKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), std::move(perm));
  } else if (const flow::UnsqueezeNode *ptr =
                 dynamic_cast<const flow::UnsqueezeNode *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    std::vector<int64_t> axes = ptr->GetAxes();
    generator = kernel::UnsqueezeKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), std::move(axes));
  } else if (const flow::UnsqueezeSubLhsScalarMulRhsScalarNode *ptr =
                 dynamic_cast<const flow::UnsqueezeSubLhsScalarMulRhsScalarNode
                                  *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    std::vector<int64_t> unsqueeze_axes = ptr->GetUnsqueezeAxes();
    Type sub_type = ptr->GetSubType(), mul_type = ptr->GetMulType();
    float64_t sub_val = ptr->GetSubVal(), mul_val = ptr->GetMulVal();
    generator = kernel::UnsqueezeSubLhsScalarMulRhsScalarKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta),
        std::move(unsqueeze_axes), sub_type, sub_val, mul_type, mul_val);
  } else if (const flow::WhereConstantCondConstantScalarYNode *ptr =
                 dynamic_cast<const flow::WhereConstantCondConstantScalarYNode
                                  *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    Tensor cond = ptr->GetCond();
    Type type = ptr->GetType();
    float64_t y = ptr->GetY();
    generator = kernel::WhereConstantCondConstantScalarYKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), std::move(cond), type,
        y);
  } else if (const flow::WhereConstantCondConstantTensorYNode *ptr =
                 dynamic_cast<const flow::WhereConstantCondConstantTensorYNode
                                  *>(node)) {
    std::shared_ptr<flow::Region> input = ptr->GetInput(),
                                  output = ptr->GetOutput();
#ifdef DEBUG
    assert(input != nullptr);
    assert(output != nullptr);
#endif
    Meta input_meta = input->GetMeta(), output_meta = output->GetMeta();
    Tensor cond = ptr->GetCond(), y = ptr->GetY();
    generator = kernel::WhereConstantCondConstantTensorYKernelGenerator::Make(
        std::move(input_meta), std::move(output_meta), std::move(cond),
        std::move(y));
  }
  return generator;
}

} // namespace worker
} // namespace cpu_transformers
