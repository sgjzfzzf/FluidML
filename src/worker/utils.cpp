#include "worker/utils.h"
#include "structure/flow/node.h"
#include "structure/kernel/add.h"
#include "structure/kernel/add_div_erf_add_mul_mul.h"
#include "structure/kernel/div.h"
#include "structure/kernel/erf.h"
#include "structure/kernel/gather.h"
#include "structure/kernel/gather_add_add.h"
#include "structure/kernel/gemm.h"
#include "structure/kernel/kernel.h"
#include "structure/kernel/layer_normalization.h"
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
#include "structure/kernel/unsqueeze_sub_mul.h"
#include "structure/kernel/where.h"

namespace cpu_transformers {
namespace worker {

std::shared_ptr<kernel::Kernel> SelectKernel(const flow::Node *node) {
  std::shared_ptr<kernel::Kernel> kernel = nullptr;
  if (const flow::AddConstantNode *ptr =
          dynamic_cast<const flow::AddConstantNode *>(node)) {
    Type type = ptr->GetType();
    float64_t constant = ptr->GetValue();
    kernel = std::make_shared<kernel::AddConstantKernel>(type, constant);
  } else if (const flow::AddCommonNode *ptr =
                 dynamic_cast<const flow::AddCommonNode *>(node)) {
    kernel = std::make_shared<kernel::AddCommonKernel>();
  } else if (const flow::AddDivErfAddMulMulNode *ptr =
                 dynamic_cast<const flow::AddDivErfAddMulMulNode *>(node)) {
    Tensor add0_weight = ptr->GetAdd0Weight();
    Type div_type = ptr->GetDivType();
    float64_t div_weight = ptr->GetDivWeight();
    Type add1_type = ptr->GetAdd1Type();
    float64_t add1_weight = ptr->GetAdd1Weight();
    Type mul1_type = ptr->GetMul1Type();
    float64_t mul1_weight = ptr->GetMul1Weight();
    kernel = std::make_shared<kernel::AddDivErfAddMulMulKernel>(
        std::move(add0_weight), div_type, div_weight, add1_type, add1_weight,
        mul1_type, mul1_weight);
  } else if (const flow::DivConstantScalarNode *ptr =
                 dynamic_cast<const flow::DivConstantScalarNode *>(node)) {
    Type type = ptr->GetType();
    float64_t constant = ptr->GetValue();
    kernel = std::make_shared<kernel::DivConstantRhsKernel>(type, constant);
  } else if (const flow::ErfNode *ptr =
                 dynamic_cast<const flow::ErfNode *>(node)) {
    kernel = std::make_shared<kernel::ErfKernel>();
  } else if (const flow::GatherConstantIndexScalarNode *ptr =
                 dynamic_cast<const flow::GatherConstantIndexScalarNode *>(
                     node)) {
    int64_t axis = ptr->GetAxis(), index = ptr->GetIndex();
    kernel =
        std::make_shared<kernel::GatherConstantIndexScalarKernel>(axis, index);
  } else if (const flow::GatherConstantDataTensorNode *ptr =
                 dynamic_cast<const flow::GatherConstantDataTensorNode *>(
                     node)) {
    Tensor data = ptr->GetData();
    kernel = std::make_shared<kernel::GatherConstantDataTensorKernel>(
        std::move(data));
  } else if (const flow::GatherConstantDataTensorAddTensorLhsAddTensorLhsNode
                 *ptr = dynamic_cast<
                     const flow::
                         GatherConstantDataTensorAddTensorLhsAddTensorLhsNode
                             *>(node)) {
    Tensor data = ptr->GetData();
    Tensor add0_weight = ptr->GetAdd0Weight();
    Tensor add1_weight = ptr->GetAdd1Weight();
    kernel = std::make_shared<
        kernel::GatherConstantDataTensorAddTensorLhsAddTensorLhsKernel>(
        std::move(data), std::move(add0_weight), std::move(add1_weight));
  } else if (const flow::GemmConstantWeightsBiasNode *ptr =
                 dynamic_cast<const flow::GemmConstantWeightsBiasNode *>(
                     node)) {
    Tensor weights = ptr->GetWeights();
    Tensor bias = ptr->GetBias();
    float64_t alpha = ptr->GetAlpha();
    float64_t beta = ptr->GetBeta();
    bool transA = ptr->GetTransA();
    bool transB = ptr->GetTransB();
    kernel = std::make_shared<kernel::GemmConstantWeightsBiasKernel>(
        alpha, beta, transA, transB, std::move(weights), std::move(bias));
  } else if (const flow::LayerNormalizationConstantScaleBiasNode *ptr =
                 dynamic_cast<
                     const flow::LayerNormalizationConstantScaleBiasNode *>(
                     node)) {
    Tensor scale = ptr->GetScale();
    Tensor bias = ptr->GetBias();
    const std::string &name = ptr->GetName();
    int64_t axis = ptr->GetAxis();
    float64_t epsilon = ptr->GetEpsilon();
    kernel =
        std::make_shared<kernel::LayerNormalizationConstantScaleBiasKernel>(
            axis, epsilon, std::move(scale), std::move(bias));
  } else if (const flow::MatMulNode *ptr =
                 dynamic_cast<const flow::MatMulNode *>(node)) {
    kernel = std::make_shared<kernel::MatMulKernel>();
  } else if (const flow::MulConstantNode *ptr =
                 dynamic_cast<const flow::MulConstantNode *>(node)) {
    Type type = ptr->GetType();
    float64_t constant = ptr->GetValue();
    kernel = std::make_shared<kernel::MulConstantKernel>(type, constant);
  } else if (const flow::MulCommonNode *ptr =
                 dynamic_cast<const flow::MulCommonNode *>(node)) {
    kernel = std::make_shared<kernel::MulCommonKernel>();
  } else if (const flow::PowNode *ptr =
                 dynamic_cast<const flow::PowNode *>(node)) {
    Type type = ptr->GetType();
    const float64_t exp = ptr->GetExp();
    kernel = std::make_shared<kernel::PowKernel>(type, exp);
  } else if (const flow::ReshapeNode *ptr =
                 dynamic_cast<const flow::ReshapeNode *>(node)) {
    kernel = std::make_shared<kernel::ReshapeKernel>();
  } else if (const flow::SoftmaxNode *ptr =
                 dynamic_cast<const flow::SoftmaxNode *>(node)) {
    const int64_t axis = ptr->GetAxis();
#ifdef DEBUG
    assert(axis >= 0);
#endif
    kernel = std::make_shared<kernel::SoftmaxKernel>(axis);
  } else if (const flow::SplitNode *ptr =
                 dynamic_cast<const flow::SplitNode *>(node)) {
    const int64_t axis = ptr->GetAxis();
#ifdef DEBUG
    assert(axis >= 0);
#endif
    kernel = std::make_shared<kernel::SplitKernel>(axis);
  } else if (const flow::SubConstantScalarLhsNode *ptr =
                 dynamic_cast<const flow::SubConstantScalarLhsNode *>(node)) {
    Type type = ptr->GetType();
    const float64_t value = ptr->GetValue();
    kernel = std::make_shared<kernel::SubConstantLhsKernel>(type, value);
  } else if (const flow::TanhNode *ptr =
                 dynamic_cast<const flow::TanhNode *>(node)) {
    kernel = std::make_shared<kernel::TanhKernel>();
  } else if (const flow::TransposeNode *ptr =
                 dynamic_cast<const flow::TransposeNode *>(node)) {
    const std::vector<int64_t> &perm = ptr->GetPerm();
    kernel = std::make_shared<kernel::TransposeKernel>(perm);
  } else if (const flow::UnsqueezeNode *ptr =
                 dynamic_cast<const flow::UnsqueezeNode *>(node)) {
    std::vector<int64_t> axes = ptr->GetAxes();
    kernel = std::make_shared<kernel::UnSqueezeKernel>(std::move(axes));
  } else if (const flow::UnsqueezeSubLhsScalarMulRhsScalarNode *ptr =
                 dynamic_cast<const flow::UnsqueezeSubLhsScalarMulRhsScalarNode
                                  *>(node)) {
    const std::vector<int64_t> &unsqueeze_axes = ptr->GetUnsqueezeAxes();
    Type sub_type = ptr->GetSubType();
    float64_t sub_val = ptr->GetSubVal();
    Type mul_type = ptr->GetMulType();
    float64_t mul_val = ptr->GetMulVal();
    kernel = std::make_shared<kernel::UnsqueezeSubLhsScalarMulRhsScalarKernel>(
        unsqueeze_axes, sub_type, sub_val, mul_type, mul_val);
  } else if (const flow::WhereConstantCondConstantScalarYNode *ptr =
                 dynamic_cast<const flow::WhereConstantCondConstantScalarYNode
                                  *>(node)) {
    Tensor cond = ptr->GetCond();
    Type type = ptr->GetType();
    float64_t y = ptr->GetY();
    kernel = std::make_shared<kernel::WhereConstantCondConstantScalarYKernel>(
        std::move(cond), type, y);
  } else if (const flow::WhereConstantCondConstantTensorYNode *ptr =
                 dynamic_cast<const flow::WhereConstantCondConstantTensorYNode
                                  *>(node)) {
    Tensor cond = ptr->GetCond();
    Tensor y = ptr->GetY();
    kernel = std::make_shared<kernel::WhereConstantCondConstantTensorYKernel>(
        std::move(cond), std::move(y));
  }
  return kernel;
}

} // namespace worker
} // namespace cpu_transformers
