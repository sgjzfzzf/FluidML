#include "structure/flow/node.h"
#include "structure/tensor/meta.h"
#include "structure/tensor/tensor.h"
#include "utils/float.h"
#include "utils/type.h"
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>
#ifdef DEBUG
#include <cassert>
#include <optional>
#endif

namespace cpu_transformers {
namespace flow {
Node::Node(std::string &&name) : name_(std::move(name)) {}

const std::string &Node::GetName() const noexcept { return name_; }

size_t Node::GetBufferSize() const noexcept { return 0; }

AddNode::AddNode(std::string &&name) : Node(std::move(name)) {}

AddConstantNode::AddConstantNode(std::string &&name,
                                 std::shared_ptr<Edge> &&input,
                                 std::shared_ptr<Edge> &&output)
    : AddNode(std::move(name)), input_(std::move(input)),
      output_(std::move(output)) {
#ifdef DEBUG
  const Meta &input_Meta = input_->GetMeta();
  const Meta &output_Meta = output_->GetMeta();
  std::optional<Meta> expected_opt =
      BroadcastShape(input_Meta, output_Meta, output_Meta.GetType());
  assert(expected_opt.has_value());
  assert(*expected_opt == output_Meta);
#endif
}

std::shared_ptr<Edge> AddConstantNode::GetInput() const noexcept {
  return input_;
}

std::shared_ptr<Edge> AddConstantNode::GetOutput() const noexcept {
  return output_;
}

AddConstantScalarNode::AddConstantScalarNode(std::string &&name, Type type,
                                             float64_t value,
                                             std::shared_ptr<Edge> &&input,
                                             std::shared_ptr<Edge> &&output)
    : AddConstantNode(std::move(name), std::move(input), std::move(output)),
      type_(type), value_(value) {}

Type AddConstantScalarNode::GetType() const noexcept { return type_; }

float64_t AddConstantScalarNode::GetValue() const noexcept { return value_; }

AddConstantTensorNode::AddConstantTensorNode(std::string &&name,
                                             Tensor &&tensor,
                                             std::shared_ptr<Edge> &&input,
                                             std::shared_ptr<Edge> &&output)
    : AddConstantNode(std::move(name), std::move(input), std::move(output)),
      tensor_(std::move(tensor)) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &tensor_meta = tensor_.GetMeta();
  const Meta &output_meta = output_->GetMeta();
  std::optional<Meta> broadcasted_meta_opt =
      BroadcastShape(input_meta, tensor_meta, output_meta.GetType());
  assert(broadcasted_meta_opt.has_value());
  assert(*broadcasted_meta_opt == output_meta);
#endif
}

Type AddConstantTensorNode::GetType() const noexcept {
  return GetTensor().GetType();
}

const Tensor &AddConstantTensorNode::GetTensor() const noexcept {
  return tensor_;
}

AddCommonNode::AddCommonNode(std::string &&name, std::shared_ptr<Edge> &&lhs,
                             std::shared_ptr<Edge> &&rhs,
                             std::shared_ptr<Edge> &&output)
    : AddNode(std::move(name)), lhs_(lhs), rhs_(rhs),
      output_(std::move(output)) {
#ifdef DEBUG
  const Meta &lhs_meta = lhs_->GetMeta();
  const Meta &rhs_meta = rhs_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  std::optional<Meta> broadcasted_meta_opt =
      BroadcastShape(lhs_meta, rhs_meta, output_meta.GetType());
  assert(broadcasted_meta_opt.has_value());
  assert(*broadcasted_meta_opt == output_meta);
#endif
}

std::shared_ptr<Edge> AddCommonNode::GetLhs() const noexcept { return lhs_; }

std::shared_ptr<Edge> AddCommonNode::GetRhs() const noexcept { return rhs_; }

std::shared_ptr<Edge> AddCommonNode::GetOutput() const noexcept {
  return output_;
}

AddDivErfAddMulMulNode::AddDivErfAddMulMulNode(
    std::string &&name, Tensor &&add0_weight, Type div_type,
    float64_t div_weight, Type add1_type, float64_t add1_weight, Type mul1_type,
    float64_t mul1_weight, std::shared_ptr<Edge> &&input,
    std::shared_ptr<Edge> &&output)
    : Node(std::move(name)), add0_weight_(add0_weight), div_type_(div_type),
      div_weight_(div_weight), add1_type_(add1_type), add1_weight_(add1_weight),
      mul1_type_(mul1_type), mul1_weight_(mul1_weight),
      input_(std::move(input)), output_(std::move(output)) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == output_meta);
#endif
}

const Tensor &AddDivErfAddMulMulNode::GetAdd0Weight() const noexcept {
  return add0_weight_;
}

Type AddDivErfAddMulMulNode::GetDivType() const noexcept { return div_type_; }

float64_t AddDivErfAddMulMulNode::GetDivWeight() const noexcept {
  return div_weight_;
}

Type AddDivErfAddMulMulNode::GetAdd1Type() const noexcept { return add1_type_; }

float64_t AddDivErfAddMulMulNode::GetAdd1Weight() const noexcept {
  return add1_weight_;
}

Type AddDivErfAddMulMulNode::GetMul1Type() const noexcept { return mul1_type_; }

float64_t AddDivErfAddMulMulNode::GetMul1Weight() const noexcept {
  return mul1_weight_;
}

std::shared_ptr<Edge> AddDivErfAddMulMulNode::GetInput() const noexcept {
  return input_;
}

std::shared_ptr<Edge> AddDivErfAddMulMulNode::GetOutput() const noexcept {
  return output_;
}

DivNode::DivNode(std::string &&name) : Node(std::move(name)) {}

DivConstantScalarNode::DivConstantScalarNode(std::string &&name, Type type,
                                             float64_t value,
                                             std::shared_ptr<Edge> &&input,
                                             std::shared_ptr<Edge> &&output)
    : DivNode(std::move(name)), type_(type), value_(value),
      input_(std::move(input)), output_(std::move(output)) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == output_meta);
#endif
}

Type DivConstantScalarNode::GetType() const noexcept { return type_; }

float64_t DivConstantScalarNode::GetValue() const noexcept { return value_; }

std::shared_ptr<Edge> DivConstantScalarNode::GetInput() const noexcept {
  return input_;
}

std::shared_ptr<Edge> DivConstantScalarNode::GetOutput() const noexcept {
  return output_;
}

ErfNode::ErfNode(std::string &&name, std::shared_ptr<Edge> &&input,
                 std::shared_ptr<Edge> &&output)
    : Node(std::move(name)), input_(std::move(input)),
      output_(std::move(output)) {}

std::shared_ptr<Edge> ErfNode::GetInput() const noexcept { return input_; }

std::shared_ptr<Edge> ErfNode::GetOutput() const noexcept { return output_; }

GatherNode::GatherNode(std::string &&name, std::shared_ptr<Edge> &&output,
                       int64_t axis)
    : Node(std::move(name)), output_(std::move(output)), axis_(axis) {
#ifdef DEBUG
  assert(output_ != nullptr);
#endif
}

std::shared_ptr<Edge> GatherNode::GetOutput() const noexcept { return output_; }

int64_t GatherNode::GetAxis() const noexcept { return axis_; }

GatherConstantIndexScalarNode::GatherConstantIndexScalarNode(
    std::string &&name, std::shared_ptr<Edge> &&lhs, int64_t rhs,
    std::shared_ptr<Edge> &&output, int64_t axis)
    : GatherNode(std::move(name), std::move(output), axis), lhs_(lhs),
      rhs_(rhs) {
#ifdef DEBUG
  const Meta &lhs_meta = lhs_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  const std::vector<int64_t> &lhs_shape = lhs_meta.GetShape();
  const std::vector<int64_t> &output_shape = output_meta.GetShape();
  const size_t lhs_shapeLen = lhs_shape.size();
  const size_t output_shape_len = output_shape.size();
  assert(output_shape_len + 1 == lhs_shapeLen);
#endif
}

std::shared_ptr<Edge> GatherConstantIndexScalarNode::GetLhs() const noexcept {
  return lhs_;
}

int64_t GatherConstantIndexScalarNode::GetRhs() const noexcept { return rhs_; }

GatherConstantDataTensorNode::GatherConstantDataTensorNode(
    std::string &&name, Tensor &&lhs, std::shared_ptr<Edge> &&rhs,
    std::shared_ptr<Edge> &&output, int64_t axis)
    : GatherNode(std::move(name), std::move(output), axis), lhs_(lhs),
      rhs_(std::move(rhs)) {
#ifdef DEBUG
  // Only the gather index of 0 is supported currently. If new cases occur, the
  // code should be updated.
  assert(axis == 0);
  const Meta &lhs_meta = lhs_.GetMeta();
  const Meta &rhs_meta = rhs_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  const std::vector<int64_t> &lhs_shape = lhs_meta.GetShape();
  const std::vector<int64_t> &rhs_shape = rhs_meta.GetShape();
  const std::vector<int64_t> &output_shape = output_meta.GetShape();
  const size_t lhs_shapeLen = lhs_shape.size();
  const size_t rhs_shape_len = rhs_shape.size();
  const size_t output_shape_len = output_shape.size();
  assert(lhs_shapeLen + rhs_shape_len - 1 == output_shape_len);
#endif
}

const Tensor &GatherConstantDataTensorNode::GetLhs() const noexcept {
  return lhs_;
}

std::shared_ptr<Edge> GatherConstantDataTensorNode::GetRhs() const noexcept {
  return rhs_;
}

GatherConstantDataTensorAddTensorLhsAddTensorLhsNode::
    GatherConstantDataTensorAddTensorLhsAddTensorLhsNode(
        std::string &&name, Tensor &&data, Tensor &&add0_weight,
        Tensor &&add1_weight, std::shared_ptr<Edge> &&input,
        std::shared_ptr<Edge> &&output)
    : Node(std::move(name)), data_(std::move(data)),
      add0_weight_(std::move(add0_weight)),
      add1_weight_(std::move(add1_weight)), input_(std::move(input)),
      output_(std::move(output)) {}

const Tensor &
GatherConstantDataTensorAddTensorLhsAddTensorLhsNode::GetData() const noexcept {
  return data_;
}

const Tensor &
GatherConstantDataTensorAddTensorLhsAddTensorLhsNode::GetAdd0Weight()
    const noexcept {
  return add0_weight_;
}

const Tensor &
GatherConstantDataTensorAddTensorLhsAddTensorLhsNode::GetAdd1Weight()
    const noexcept {
  return add1_weight_;
}

std::shared_ptr<Edge>
GatherConstantDataTensorAddTensorLhsAddTensorLhsNode::GetInput()
    const noexcept {
  return input_;
}

std::shared_ptr<Edge>
GatherConstantDataTensorAddTensorLhsAddTensorLhsNode::GetOutput()
    const noexcept {
  return output_;
}

GemmNode::GemmNode(std::string &&name, float64_t alpha = GemmNode::kAlpha,
                   float64_t beta = GemmNode::kBeta,
                   bool transA = GemmNode::kTransA,
                   bool transB = GemmNode::kTransB)
    : Node(std::move(name)), alpha_(alpha), beta_(beta), transA_(transA),
      transB_(transB) {}

float64_t GemmNode::GetAlpha() const noexcept { return alpha_; }

float64_t GemmNode::GetBeta() const noexcept { return beta_; }

bool GemmNode::GetTransA() const noexcept { return transA_; }

bool GemmNode::GetTransB() const noexcept { return transB_; }

GemmConstantWeightsBiasNode::GemmConstantWeightsBiasNode(
    std::string &&name, std::shared_ptr<Edge> &&input, Tensor &&weights,
    Tensor &&bias, std::shared_ptr<Edge> &&output,
    float64_t alpha = GemmNode::kAlpha, float64_t beta = GemmNode::kBeta,
    bool transA = GemmNode::kTransA, bool transB = GemmNode::kTransB)
    : GemmNode(std::move(name), alpha, beta, transA, transB),
      input_(std::move(input)), weights_(std::move(weights)),
      bias_(std::move(bias)), output_(std::move(output)) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  const std::vector<int64_t> &input_shape = input_meta.GetShape();
  const std::vector<int64_t> &weights_shape = weights_.GetShape();
  const std::vector<int64_t> &bias_shape = bias_.GetShape();
  const std::vector<int64_t> &output_shape = output_meta.GetShape();
  assert(input_shape.size() == 2);
  assert(weights_shape.size() == 2);
  size_t bias_shape_len = bias_shape.size();
  assert(bias_shape_len == 1 || bias_shape_len == 2);
  assert(output_shape.size() == 2);
  size_t m = std::lround(input_shape[0]);
  size_t k = std::lround(input_shape[1]);
  size_t n = std::lround(weights_shape[1]);
  assert(weights_shape[0] == k);
  if (bias_shape_len == 1) {
    assert(bias_shape[0] == n);
  } else {
    assert(bias_shape[0] == m);
    assert(bias_shape[1] == n);
  }
  assert(output_shape[0] == m);
  assert(output_shape[1] == n);
#endif
}

std::shared_ptr<Edge> GemmConstantWeightsBiasNode::GetInput() const noexcept {
  return input_;
}

const Tensor &GemmConstantWeightsBiasNode::GetWeights() const noexcept {
  return weights_;
}

const Tensor &GemmConstantWeightsBiasNode::GetBias() const noexcept {
  return bias_;
}

std::shared_ptr<Edge> GemmConstantWeightsBiasNode::GetOutput() const noexcept {
  return output_;
}

LayerNormalizationNode::LayerNormalizationNode(
    std::string &&name, int64_t axis = LayerNormalizationNode::kAxis,
    float64_t epsilon = LayerNormalizationNode::kEpsilon)
    : Node(std::move(name)), axis_(axis), epsilon_(epsilon) {}

int64_t LayerNormalizationNode::GetAxis() const noexcept {
  return axis_ >= 0 ? axis_ : GetMeta().GetShape().size() + axis_;
}

float64_t LayerNormalizationNode::GetEpsilon() const noexcept {
  return epsilon_;
}

LayerNormalizationConstantScaleBiasNode::
    LayerNormalizationConstantScaleBiasNode(std::string &&name,
                                            std::shared_ptr<Edge> &&input,
                                            Tensor &&scale, Tensor &&bias,
                                            std::shared_ptr<Edge> &&output,
                                            int64_t axis, float64_t epsilon)
    : LayerNormalizationNode(std::move(name), axis, epsilon),
      input_(std::move(input)), scale_(std::move(scale)),
      bias_(std::move(bias)), output_(std::move(output)) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == output_meta);
#endif
}

size_t LayerNormalizationConstantScaleBiasNode::GetBufferSize() const noexcept {
  const Meta &meta = GetMeta();
  Type type = meta.GetType();
  std::vector<int64_t> shape = meta.GetShape();
  const size_t axis = GetAxis();
#ifdef DEBUG
  assert(axis >= 0 && axis < shape.size());
#endif
  shape[axis] = 1;
  const int64_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                       std::multiplies<int64_t>()) *
                       GetSizeFromType(type);
  return size;
}

const Meta &LayerNormalizationConstantScaleBiasNode::GetMeta() const noexcept {
  return input_->GetMeta();
}

std::shared_ptr<Edge>
LayerNormalizationConstantScaleBiasNode::GetInput() const noexcept {
  return input_;
}

const Tensor &
LayerNormalizationConstantScaleBiasNode::GetScale() const noexcept {
  return scale_;
}

const Tensor &
LayerNormalizationConstantScaleBiasNode::GetBias() const noexcept {
  return bias_;
}

std::shared_ptr<Edge>
LayerNormalizationConstantScaleBiasNode::GetOutput() const noexcept {
  return output_;
}

MatMulNode::MatMulNode(std::string &&name) : Node(std::move(name)) {}

MatMulConstantLhsNode::MatMulConstantLhsNode(std::string &&name, Tensor &&lhs,
                                             std::shared_ptr<Edge> &&rhs,
                                             std::shared_ptr<Edge> &&output)
    : MatMulNode(std::move(name)), lhs_(std::move(lhs)), rhs_(std::move(rhs)),
      output_(std::move(output)) {
#ifdef DEBUG
  const Meta &rhs_meta = rhs_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  const std::vector<int64_t> &lhs_shape = lhs_.GetShape();
  const std::vector<int64_t> &rhs_shape = rhs_meta.GetShape();
  const std::vector<int64_t> &output_shape = output_meta.GetShape();
  size_t lhs_shapeLen = lhs_shape.size();
  size_t rhs_shape_len = rhs_shape.size();
  size_t output_shape_len = output_shape.size();
  assert(lhs_shapeLen >= 2);
  assert(rhs_shape_len >= 2);
  assert(output_shape_len >= 2);
  size_t m = lhs_shape[lhs_shapeLen - 2];
  size_t k = lhs_shape[lhs_shapeLen - 1];
  size_t n = rhs_shape[rhs_shape_len - 1];
  assert(rhs_shape[rhs_shape_len - 2] == k);
  assert(output_shape[output_shape_len - 2] == m);
  assert(output_shape[output_shape_len - 1] == n);
  std::optional<Meta> expected_output_meta_opt =
      BroadcastMatMulShape(lhs_.GetMeta(), rhs_meta, output_meta.GetType());
  assert(expected_output_meta_opt.has_value());
  assert(*expected_output_meta_opt == output_meta);
#endif
}

const Tensor &MatMulConstantLhsNode::GetLhs() const noexcept { return lhs_; }

std::shared_ptr<Edge> MatMulConstantLhsNode::GetRhs() const noexcept {
  return rhs_;
}

std::shared_ptr<Edge> MatMulConstantLhsNode::GetOutput() const noexcept {
  return output_;
}

MatMulConstantRhsNode::MatMulConstantRhsNode(std::string &&name,
                                             std::shared_ptr<Edge> &&lhs,
                                             Tensor &&rhs,
                                             std::shared_ptr<Edge> &&output)
    : MatMulNode(std::move(name)), lhs_(std::move(lhs)), rhs_(std::move(rhs)),
      output_(std::move(output)) {
#ifdef DEBUG
  const Meta &lhs_meta = lhs_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  const std::vector<int64_t> &lhs_shape = lhs_meta.GetShape();
  const std::vector<int64_t> &rhs_shape = rhs_.GetShape();
  const std::vector<int64_t> &output_shape = output_meta.GetShape();
  size_t lhs_shapeLen = lhs_shape.size();
  size_t rhs_shape_len = rhs_shape.size();
  size_t output_shape_len = output_shape.size();
  assert(lhs_shapeLen >= 2);
  assert(rhs_shape_len >= 2);
  assert(output_shape_len >= 2);
  size_t m = lhs_shape[lhs_shapeLen - 2];
  size_t k = lhs_shape[lhs_shapeLen - 1];
  size_t n = rhs_shape[rhs_shape_len - 1];
  assert(rhs_shape[rhs_shape_len - 2] == k);
  assert(output_shape[output_shape_len - 2] == m);
  assert(output_shape[output_shape_len - 1] == n);
  std::optional<Meta> expected_output_meta_opt =
      BroadcastMatMulShape(lhs_meta, rhs_.GetMeta(), output_meta.GetType());
  assert(expected_output_meta_opt.has_value());
  assert(*expected_output_meta_opt == output_meta);
#endif
}

std::shared_ptr<Edge> MatMulConstantRhsNode::GetLhs() const noexcept {
  return lhs_;
}

const Tensor &MatMulConstantRhsNode::GetRhs() const noexcept { return rhs_; }

std::shared_ptr<Edge> MatMulConstantRhsNode::GetOutput() const noexcept {
  return output_;
}

MatMulCommonNode::MatMulCommonNode(std::string &&name,
                                   std::shared_ptr<Edge> &&lhs,
                                   std::shared_ptr<Edge> &&rhs,
                                   std::shared_ptr<Edge> &&output)
    : MatMulNode(std::move(name)), lhs_(std::move(lhs)), rhs_(std::move(rhs)),
      output_(std::move(output)) {
#ifdef DEBUG
  const Meta &lhs_meta = lhs_->GetMeta();
  const Meta &rhs_meta = rhs_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  const std::vector<int64_t> &lhs_shape = lhs_meta.GetShape();
  const std::vector<int64_t> &rhs_shape = rhs_meta.GetShape();
  const std::vector<int64_t> &output_shape = output_meta.GetShape();
  size_t lhs_shapeLen = lhs_shape.size();
  size_t rhs_shape_len = rhs_shape.size();
  size_t output_shape_len = output_shape.size();
  assert(lhs_shapeLen >= 2);
  assert(rhs_shape_len >= 2);
  assert(output_shape_len >= 2);
  size_t m = lhs_shape[lhs_shapeLen - 2];
  size_t k = lhs_shape[lhs_shapeLen - 1];
  size_t n = rhs_shape[rhs_shape_len - 1];
  assert(rhs_shape[rhs_shape_len - 2] == k);
  assert(output_shape[output_shape_len - 2] == m);
  assert(output_shape[output_shape_len - 1] == n);
  std::optional<Meta> expected_output_meta_opt =
      BroadcastMatMulShape(lhs_meta, rhs_meta, output_meta.GetType());
  assert(expected_output_meta_opt.has_value());
  assert(*expected_output_meta_opt == output_meta);
#endif
}

std::shared_ptr<Edge> MatMulCommonNode::GetLhs() const noexcept { return lhs_; }

std::shared_ptr<Edge> MatMulCommonNode::GetRhs() const noexcept { return rhs_; }

std::shared_ptr<Edge> MatMulCommonNode::GetOutput() const noexcept {
  return output_;
}

MulNode::MulNode(std::string &&name) : Node(std::move(name)) {}

MulConstantNode::MulConstantNode(std::string &&name,
                                 std::shared_ptr<Edge> &&input,
                                 std::shared_ptr<Edge> &&output)
    : MulNode(std::move(name)), input_(std::move(input)),
      output_(std::move(output)) {}

std::shared_ptr<Edge> MulConstantNode::GetInput() const noexcept {
  return input_;
}

std::shared_ptr<Edge> MulConstantNode::GetOutput() const noexcept {
  return output_;
}

MulConstantScalarNode::MulConstantScalarNode(std::string &&name,
                                             std::shared_ptr<Edge> &&input,
                                             Type type, float64_t value,
                                             std::shared_ptr<Edge> &&output)
    : MulConstantNode(std::move(name), std::move(input), std::move(output)),
      type_(type), value_(value) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == output_meta);
#endif
}

Type MulConstantScalarNode::GetType() const noexcept { return type_; }

float64_t MulConstantScalarNode::GetValue() const noexcept { return value_; }

MulConstantTensorNode::MulConstantTensorNode(std::string &&name,
                                             std::shared_ptr<Edge> &&input,
                                             Tensor &&tensor,
                                             std::shared_ptr<Edge> &&output)
    : MulConstantNode(std::move(name), std::move(input), std::move(output)),
      tensor_(std::move(tensor)) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == tensor_.GetMeta());
  assert(input_meta == output_meta);
#endif
}

const Tensor &MulConstantTensorNode::GetTensor() const noexcept {
  return tensor_;
}

MulCommonNode::MulCommonNode(std::string &&name, std::shared_ptr<Edge> &&lhs,
                             std::shared_ptr<Edge> &&rhs,
                             std::shared_ptr<Edge> &&output)
    : MulNode(std::move(name)), lhs_(std::move(lhs)), rhs_(std::move(rhs)),
      output_(std::move(output)) {}

std::shared_ptr<Edge> MulCommonNode::GetLhs() const noexcept { return lhs_; }

std::shared_ptr<Edge> MulCommonNode::GetRhs() const noexcept { return rhs_; }

std::shared_ptr<Edge> MulCommonNode::GetOutput() const noexcept {
  return output_;
}

PowNode::PowNode(std::string &&name, std::shared_ptr<Edge> &&input, Type type,
                 float64_t exp, std::shared_ptr<Edge> &&output)
    : Node(std::move(name)), input_(std::move(input)), type_(type), exp_(exp),
      output_(std::move(output)) {}

std::shared_ptr<Edge> PowNode::GetInput() const noexcept { return input_; }

Type PowNode::GetType() const noexcept { return type_; }

float64_t PowNode::GetExp() const noexcept { return exp_; }

std::shared_ptr<Edge> PowNode::GetOutput() const noexcept { return output_; }

ReshapeNode::ReshapeNode(std::string &&name, std::shared_ptr<Edge> &&input,
                         std::shared_ptr<Edge> &&output)
    : Node(std::move(name)), input_(std::move(input)),
      output_(std::move(output)) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  const std::vector<int64_t> &input_shape = input_meta.GetShape();
  const std::vector<int64_t> &output_shape = output_meta.GetShape();
  assert(std::accumulate(input_shape.begin(), input_shape.end(), 1,
                         std::multiplies<size_t>()) ==
         std::accumulate(output_shape.begin(), output_shape.end(), 1,
                         std::multiplies<size_t>()));
#endif
}

std::shared_ptr<Edge> ReshapeNode::GetInput() const noexcept { return input_; }

std::shared_ptr<Edge> ReshapeNode::GetOutput() const noexcept {
  return output_;
}

SoftmaxNode::SoftmaxNode(std::string &&name, std::shared_ptr<Edge> &&input,
                         std::shared_ptr<Edge> &&output, int64_t axis)
    : Node(std::move(name)), input_(std::move(input)),
      output_(std::move(output)), axis_(axis) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == output_meta);
#endif
}

size_t SoftmaxNode::GetBufferSize() const noexcept {
  const Meta &meta = input_->GetMeta();
  Type type = meta.GetType();
  std::vector<int64_t> shape = meta.GetShape();
  const size_t axis = GetAxis();
#ifdef DEBUG
  assert(axis >= 0 && axis < shape.size());
#endif
  shape[axis] = 1;
  const int64_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                       std::multiplies<int64_t>()) *
                       GetSizeFromType(type);
  return size;
}

std::shared_ptr<Edge> SoftmaxNode::GetInput() const noexcept { return input_; }

std::shared_ptr<Edge> SoftmaxNode::GetOutput() const noexcept {
  return output_;
}

int64_t SoftmaxNode::GetAxis() const noexcept {
  return axis_ >= 0 ? axis_ : GetMeta().GetShape().size() + axis_;
}

const Meta &SoftmaxNode::GetMeta() const noexcept {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == output_meta);
#endif
  return input_->GetMeta();
}

SplitNode::SplitNode(std::string &&name, std::shared_ptr<Edge> &&input,
                     std::vector<std::shared_ptr<Edge>> &&outputs, int64_t axis)
    : Node(std::move(name)), input_(std::move(input)),
      outputs_(std::move(outputs)), axis_(axis) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const std::vector<int64_t> &input_shape = input_meta.GetShape();
  size_t input_shape_len = input_shape.size();
  assert(axis < input_shape_len);
  size_t sum = 0;
  for (const std::shared_ptr<Edge> &output : outputs_) {
    const Meta &output_meta = output->GetMeta();
    const std::vector<int64_t> &output_shape = output_meta.GetShape();
    size_t output_shape_len = output_shape.size();
    assert(input_shape_len == output_shape_len);
    int64_t axis = GetAxis();
    for (size_t i = 0; i < output_shape_len; ++i) {
      if (axis == i) {
        sum += output_shape[i];
      } else {
        assert(input_shape[i] == output_shape[i]);
      }
    }
  }
  assert(sum == input_shape[axis]);
#endif
}

std::shared_ptr<Edge> SplitNode::GetInput() const noexcept { return input_; }

const std::vector<std::shared_ptr<Edge>> &
SplitNode::GetOutputs() const noexcept {
  return outputs_;
}

int64_t SplitNode::GetAxis() const noexcept {
  return axis_ >= 0 ? axis_ : GetMeta().GetShape().size() + axis_;
}

const Meta &SplitNode::GetMeta() const noexcept { return input_->GetMeta(); }

SubNode::SubNode(std::string &&name) : Node(std::move(name)) {}

SubConstantScalarLhsNode::SubConstantScalarLhsNode(
    std::string &&name, std::shared_ptr<Edge> &&input, Type type,
    float64_t value, std::shared_ptr<Edge> &&output)
    : SubNode(std::move(name)), input_(std::move(input)), type_(type),
      value_(value), output_(std::move(output)) {}

std::shared_ptr<Edge> SubConstantScalarLhsNode::GetInput() const noexcept {
  return input_;
}

Type SubConstantScalarLhsNode::GetType() const noexcept { return type_; }

float64_t SubConstantScalarLhsNode::GetValue() const noexcept { return value_; }

std::shared_ptr<Edge> SubConstantScalarLhsNode::GetOutput() const noexcept {
  return output_;
}

TanhNode::TanhNode(std::string &&name, std::shared_ptr<Edge> &&input,
                   std::shared_ptr<Edge> &&output)
    : Node(std::move(name)), input_(std::move(input)),
      output_(std::move(output)) {}

std::shared_ptr<Edge> TanhNode::GetInput() const noexcept { return input_; }

std::shared_ptr<Edge> TanhNode::GetOutput() const noexcept { return output_; }

TransposeNode::TransposeNode(std::string &&name, std::shared_ptr<Edge> &&input,
                             std::shared_ptr<Edge> &&output,
                             std::vector<int64_t> &&perm)
    : Node(std::move(name)), input_(std::move(input)),
      output_(std::move(output)), perm_(std::move(perm)) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  const std::vector<int64_t> &input_shape = input_meta.GetShape();
  const std::vector<int64_t> &output_shape = output_meta.GetShape();
  size_t input_shape_len = input_shape.size();
  size_t output_shape_len = output_shape.size();
  assert(input_shape_len == output_shape_len);
  std::vector<int64_t> expected_output_shape(input_shape_len, 0);
  for (size_t i = 0; i < input_shape_len; ++i) {
    int64_t index = perm_[i];
    assert(index >= 0 && index < input_shape_len);
    expected_output_shape[i] = input_shape[index];
  }
  assert(expected_output_shape == output_shape);
#endif
}

std::shared_ptr<Edge> TransposeNode::GetInput() const noexcept {
  return input_;
}

std::shared_ptr<Edge> TransposeNode::GetOutput() const noexcept {
  return output_;
}

const std::vector<int64_t> &TransposeNode::GetPerm() const noexcept {
  return perm_;
}

UnsqueezeNode::UnsqueezeNode(std::string &&name, std::shared_ptr<Edge> &&input,
                             std::shared_ptr<Edge> &&output,
                             std::vector<int64_t> &&axes)
    : Node(std::move(name)), input_(std::move(input)),
      output_(std::move(output)), axes_(std::move(axes)) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  const std::vector<int64_t> &input_shape = input_meta.GetShape();
  const std::vector<int64_t> &output_shape = output_meta.GetShape();
  size_t input_shape_len = input_shape.size();
  size_t output_shape_len = output_shape.size();
  assert(output_shape_len == input_shape_len + axes_.size());
  for (size_t i = 0, j = 0; i < output_shape_len; ++i) {
    if (j < axes_.size() && axes_[j] == i) {
      ++j;
    } else {
      assert(input_shape[i - j] == output_shape[i]);
    }
  }
#endif
}

std::shared_ptr<Edge> UnsqueezeNode::GetInput() const noexcept {
  return input_;
}

std::shared_ptr<Edge> UnsqueezeNode::GetOutput() const noexcept {
  return output_;
}

const std::vector<int64_t> &UnsqueezeNode::GetAxes() const noexcept {
  return axes_;
}

UnsqueezeSubLhsScalarMulRhsScalarNode::UnsqueezeSubLhsScalarMulRhsScalarNode(
    std::string &&name, std::shared_ptr<Edge> &&input,
    std::shared_ptr<Edge> &&output, std::vector<int64_t> &&unsqueeze_axes,
    Type sub_type, float64_t sub_val, Type mul_type, float64_t mul_val)
    : Node(std::move(name)), input_(std::move(input)),
      output_(std::move(output)), unsqueeze_axes_(std::move(unsqueeze_axes)),
      sub_type_(sub_type), sub_val_(sub_val), mul_type_(mul_type),
      mul_val_(mul_val) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  const std::vector<int64_t> &input_shape = input_meta.GetShape();
  const std::vector<int64_t> &output_shape = output_meta.GetShape();
  size_t input_shape_len = input_shape.size();
  size_t output_shape_len = output_shape.size();
  assert(output_shape_len == input_shape_len + unsqueeze_axes_.size());
  for (size_t i = 0, j = 0; i < output_shape_len; ++i) {
    if (j < unsqueeze_axes_.size() && unsqueeze_axes_[j] == i) {
      ++j;
    } else {
      assert(input_shape[i - j] == output_shape[i]);
    }
  }
#endif
}

std::shared_ptr<Edge>
UnsqueezeSubLhsScalarMulRhsScalarNode::GetInput() const noexcept {
  return input_;
}

std::shared_ptr<Edge>
UnsqueezeSubLhsScalarMulRhsScalarNode::GetOutput() const noexcept {
  return output_;
}

const std::vector<int64_t> &
UnsqueezeSubLhsScalarMulRhsScalarNode::GetUnsqueezeAxes() const noexcept {
  return unsqueeze_axes_;
}

Type UnsqueezeSubLhsScalarMulRhsScalarNode::GetSubType() const noexcept {
  return sub_type_;
}

float64_t UnsqueezeSubLhsScalarMulRhsScalarNode::GetSubVal() const noexcept {
  return sub_val_;
}

Type UnsqueezeSubLhsScalarMulRhsScalarNode::GetMulType() const noexcept {
  return mul_type_;
}

float64_t UnsqueezeSubLhsScalarMulRhsScalarNode::GetMulVal() const noexcept {
  return mul_val_;
}

WhereNode::WhereNode(std::string &&name) : Node(std::move(name)) {}

WhereConstantCondConstantScalarYNode::WhereConstantCondConstantScalarYNode(
    std::string &&name, Tensor &&cond, std::shared_ptr<Edge> &&x, Type type,
    float64_t y, std::shared_ptr<Edge> &&output)
    : WhereNode(std::move(name)), cond_(std::move(cond)), x_(std::move(x)),
      type_(type), y_(y), output_(std::move(output)) {
#ifdef DEBUG
  assert(cond_.GetType() == Type::BOOL);
  const Meta &x_meta = x_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  std::optional<Meta> broadcasted_meta_opt =
      BroadcastShape(cond_.GetMeta(), x_meta, output_meta.GetType());
  assert(broadcasted_meta_opt.has_value());
  assert(*broadcasted_meta_opt == output_meta);
#endif
}

const Tensor &WhereConstantCondConstantScalarYNode::GetCond() const noexcept {
  return cond_;
}

std::shared_ptr<Edge>
WhereConstantCondConstantScalarYNode::GetX() const noexcept {
  return x_;
}

Type WhereConstantCondConstantScalarYNode::GetType() const noexcept {
  return type_;
}

float64_t WhereConstantCondConstantScalarYNode::GetY() const noexcept {
  return y_;
}

std::shared_ptr<Edge>
WhereConstantCondConstantScalarYNode::GetOutput() const noexcept {
  return output_;
}

WhereConstantCondConstantTensorYNode::WhereConstantCondConstantTensorYNode(
    std::string &&name, Tensor &&cond, std::shared_ptr<Edge> &&x, Tensor &&y,
    std::shared_ptr<Edge> &&output)
    : WhereNode(std::move(name)), cond_(std::move(cond)), x_(std::move(x)),
      y_(std::move(y)), output_(std::move(output)) {
#ifdef DEBUG
  assert(cond_.GetType() == Type::BOOL);
  const Meta &x_meta = x_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(x_meta == cond_.GetMeta());
  assert(x_meta == y_.GetMeta());
  assert(x_meta == output_meta);
#endif
}

const Tensor &WhereConstantCondConstantTensorYNode::GetCond() const noexcept {
  return cond_;
}

std::shared_ptr<Edge>
WhereConstantCondConstantTensorYNode::GetX() const noexcept {
  return x_;
}

const Tensor &WhereConstantCondConstantTensorYNode::GetY() const noexcept {
  return y_;
}

std::shared_ptr<Edge>
WhereConstantCondConstantTensorYNode::GetOutput() const noexcept {
  return output_;
}

} // namespace flow
} // namespace cpu_transformers
