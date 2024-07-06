#include "structure/flow/node.h"
#include "structure/flow/region.h"
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

SingleInputNode::SingleInputNode(std::string &&name,
                                 std::shared_ptr<Region> &&input,
                                 std::shared_ptr<Region> &&output)
    : Node(std::move(name)), input_(std::move(input)),
      output_(std::move(output)) {}

std::shared_ptr<Node> SingleInputNode::CloneAsNode() const {
  return CloneAsSingleInputNode();
}

std::shared_ptr<Region> SingleInputNode::GetInput() const noexcept {
  return input_;
}

std::shared_ptr<Region> SingleInputNode::GetOutput() const noexcept {
  return output_;
}

const std::string &SingleInputNode::GetInputAsString() const noexcept {
  std::shared_ptr<Region> input = GetInput();
#ifdef DEBUG
  assert(input != nullptr);
#endif
  return input->GetName();
}

const std::string &SingleInputNode::GetOutputAsString() const noexcept {
  std::shared_ptr<Region> output = GetOutput();
#ifdef DEBUG
  assert(output != nullptr);
#endif
  return output->GetName();
}

void SingleInputNode::SetInput(std::shared_ptr<Region> &&input) noexcept {
  input_ = std::move(input);
}

void SingleInputNode::SetOutput(std::shared_ptr<Region> &&output) noexcept {
  output_ = std::move(output);
}

DoubleInputsNode::DoubleInputsNode(std::string &&name,
                                   std::shared_ptr<Region> &&lhs,
                                   std::shared_ptr<Region> &&rhs,
                                   std::shared_ptr<Region> &&output)
    : Node(std::move(name)), lhs_(std::move(lhs)), rhs_(std::move(rhs)),
      output_(std::move(output)) {}

std::shared_ptr<Node> DoubleInputsNode::CloneAsNode() const {
  return CloneAsDoubleInputsNode();
}

std::shared_ptr<Region> DoubleInputsNode::GetLhs() const noexcept {
  return lhs_;
}

std::shared_ptr<Region> DoubleInputsNode::GetRhs() const noexcept {
  return rhs_;
}

std::shared_ptr<Region> DoubleInputsNode::GetOutput() const noexcept {
  return output_;
}

const std::string &DoubleInputsNode::GetLhsAsString() const noexcept {
  std::shared_ptr<Region> lhs = GetLhs();
#ifdef DEBUG
  assert(lhs != nullptr);
#endif
  return lhs->GetName();
}

const std::string &DoubleInputsNode::GetRhsAsString() const noexcept {
  std::shared_ptr<Region> rhs = GetRhs();
#ifdef DEBUG
  assert(rhs != nullptr);
#endif
  return rhs->GetName();
}

const std::string &DoubleInputsNode::GetOutputAsString() const noexcept {
  std::shared_ptr<Region> output = GetOutput();
#ifdef DEBUG
  assert(output != nullptr);
#endif
  return output->GetName();
}

void DoubleInputsNode::SetLhs(std::shared_ptr<Region> &&lhs) noexcept {
  lhs_ = std::move(lhs);
}

void DoubleInputsNode::SetRhs(std::shared_ptr<Region> &&rhs) noexcept {
  rhs_ = std::move(rhs);
}

void DoubleInputsNode::SetOutput(std::shared_ptr<Region> &&output) noexcept {
  output_ = std::move(output);
}

SingleInputWithoutBufferNode::SingleInputWithoutBufferNode(
    std::string &&name, std::shared_ptr<Region> &&input,
    std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      SingleInputNode(std::move(name), std::move(input), std::move(output)) {}

std::shared_ptr<SingleInputNode>
SingleInputWithoutBufferNode::CloneAsSingleInputNode() const {
  return CloneAsSingleInputWithoutBufferNode();
}

SingleInputWithBufferNode::SingleInputWithBufferNode(
    std::string &&name, std::shared_ptr<Region> &&input,
    std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      SingleInputNode(std::move(name), std::move(input), std::move(output)) {}

std::shared_ptr<SingleInputNode>
SingleInputWithBufferNode::CloneAsSingleInputNode() const {
  return CloneAsSingleInputWithBufferNode();
}

DoubleInputsWithoutBufferNode::DoubleInputsWithoutBufferNode(
    std::string &&name, std::shared_ptr<Region> &&lhs,
    std::shared_ptr<Region> &&rhs, std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      DoubleInputsNode(std::move(name), std::move(lhs), std::move(rhs),
                       std::move(output)) {}

std::shared_ptr<DoubleInputsNode>
DoubleInputsWithoutBufferNode::CloneAsDoubleInputsNode() const {
  return CloneAsDoubleInputsWithoutBufferNode();
}

DoubleInputsWithBufferNode::DoubleInputsWithBufferNode(
    std::string &&name, std::shared_ptr<Region> &&lhs,
    std::shared_ptr<Region> &&rhs, std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      DoubleInputsNode(std::move(name), std::move(lhs), std::move(rhs),
                       std::move(output)) {}

std::shared_ptr<DoubleInputsNode>
DoubleInputsWithBufferNode::CloneAsDoubleInputsNode() const {
  return CloneAsDoubleInputsWithBufferNode();
}

AddNode::AddNode(std::string &&name) : Node(std::move(name)) {}

AddConstantNode::AddConstantNode(std::string &&name,
                                 std::shared_ptr<Region> &&input,
                                 std::shared_ptr<Region> &&output)
    : AddNode(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)) {
#ifdef DEBUG
  const Meta &input_Meta = input_->GetMeta();
  const Meta &output_Meta = output_->GetMeta();
  std::optional<Meta> expected_opt =
      BroadcastShape(input_Meta, output_Meta, output_Meta.GetType());
  assert(expected_opt.has_value());
  assert(*expected_opt == output_Meta);
#endif
}

std::shared_ptr<SingleInputWithoutBufferNode>
AddConstantNode::CloneAsSingleInputWithoutBufferNode() const {
  return CloneAsAddConstantNode();
}

AddConstantScalarNode::AddConstantScalarNode(std::string &&name, Type type,
                                             float64_t value,
                                             std::shared_ptr<Region> &&input,
                                             std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      AddConstantNode(std::move(name), std::move(input), std::move(output)),
      type_(type), value_(value) {}

std::shared_ptr<AddConstantNode>
AddConstantScalarNode::CloneAsAddConstantNode() const {
  return Clone();
}

std::shared_ptr<AddConstantScalarNode> AddConstantScalarNode::Clone() const {
  std::string name = GetName();
  return std::make_shared<AddConstantScalarNode>(
      std::move(name), GetType(), GetValue(), GetInput(), GetOutput());
}

Type AddConstantScalarNode::GetType() const noexcept { return type_; }

float64_t AddConstantScalarNode::GetValue() const noexcept { return value_; }

AddConstantTensorNode::AddConstantTensorNode(std::string &&name,
                                             Tensor &&tensor,
                                             std::shared_ptr<Region> &&input,
                                             std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      AddConstantNode(std::move(name), std::move(input), std::move(output)),
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

std::shared_ptr<AddConstantNode>
AddConstantTensorNode::CloneAsAddConstantNode() const {
  return Clone();
}

std::shared_ptr<AddConstantTensorNode> AddConstantTensorNode::Clone() const {
  std::string name = GetName();
  Tensor tensor = GetTensor();
  return std::make_shared<AddConstantTensorNode>(
      std::move(name), std::move(tensor), GetInput(), GetOutput());
}

Type AddConstantTensorNode::GetType() const noexcept {
  return GetTensor().GetType();
}

const Tensor &AddConstantTensorNode::GetTensor() const noexcept {
  return tensor_;
}

AddCommonNode::AddCommonNode(std::string &&name, std::shared_ptr<Region> &&lhs,
                             std::shared_ptr<Region> &&rhs,
                             std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      DoubleInputsWithoutBufferNode(std::move(name), std::move(lhs),
                                    std::move(rhs), std::move(output)),
      AddNode(std::move(name)) {
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

std::shared_ptr<DoubleInputsWithoutBufferNode>
AddCommonNode::CloneAsDoubleInputsWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<AddCommonNode> AddCommonNode::Clone() const {
  std::string name = GetName();
  return std::make_shared<AddCommonNode>(std::move(name), GetLhs(), GetRhs(),
                                         GetOutput());
}

AddDivErfAddMulMulNode::AddDivErfAddMulMulNode(
    std::string &&name, Tensor &&add0_weight, Type div_type,
    float64_t div_weight, Type add1_type, float64_t add1_weight, Type mul1_type,
    float64_t mul1_weight, std::shared_ptr<Region> &&input,
    std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      add0_weight_(add0_weight), div_type_(div_type), div_weight_(div_weight),
      add1_type_(add1_type), add1_weight_(add1_weight), mul1_type_(mul1_type),
      mul1_weight_(mul1_weight) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == output_meta);
#endif
}

std::shared_ptr<SingleInputWithoutBufferNode>
AddDivErfAddMulMulNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<AddDivErfAddMulMulNode> AddDivErfAddMulMulNode::Clone() const {
  std::string name = GetName();
  Tensor add0_weight = GetAdd0Weight();
  return std::make_shared<AddDivErfAddMulMulNode>(
      std::move(name), std::move(add0_weight), GetDivType(), GetDivWeight(),
      GetAdd1Type(), GetAdd1Weight(), GetMul1Type(), GetMul1Weight(),
      GetInput(), GetOutput());
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

DivNode::DivNode(std::string &&name) : Node(std::move(name)) {}

DivConstantScalarNode::DivConstantScalarNode(std::string &&name, Type type,
                                             float64_t value,
                                             std::shared_ptr<Region> &&input,
                                             std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      DivNode(std::move(name)), type_(type), value_(value) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == output_meta);
#endif
}

std::shared_ptr<SingleInputWithoutBufferNode>
DivConstantScalarNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<DivConstantScalarNode> DivConstantScalarNode::Clone() const {
  std::string name = GetName();
  return std::make_shared<DivConstantScalarNode>(
      std::move(name), GetType(), GetValue(), GetInput(), GetOutput());
}

Type DivConstantScalarNode::GetType() const noexcept { return type_; }

float64_t DivConstantScalarNode::GetValue() const noexcept { return value_; }

ErfNode::ErfNode(std::string &&name, std::shared_ptr<Region> &&input,
                 std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)) {}

std::shared_ptr<SingleInputWithoutBufferNode>
ErfNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<ErfNode> ErfNode::Clone() const {
  std::string name = GetName();
  return std::make_shared<ErfNode>(std::move(name), GetInput(), GetOutput());
}

GatherNode::GatherNode(std::string &&name) : Node(std::move(name)) {}

GatherConstantIndexScalarNode::GatherConstantIndexScalarNode(
    std::string &&name, std::shared_ptr<Region> &&input,
    std::shared_ptr<Region> &&output, int64_t index, int64_t axis)
    : Node(std::move(name)), GatherNode(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      index_(index), axis_(axis) {
#ifdef DEBUG
  assert(input_ != nullptr);
  assert(output_ != nullptr);
  const Meta &lhs_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  const std::vector<int64_t> &lhs_shape = lhs_meta.GetShape();
  const std::vector<int64_t> &output_shape = output_meta.GetShape();
  const size_t lhs_shapeLen = lhs_shape.size();
  const size_t output_shape_len = output_shape.size();
  assert(output_shape_len + 1 == lhs_shapeLen);
#endif
}

std::shared_ptr<SingleInputWithoutBufferNode>
GatherConstantIndexScalarNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<GatherConstantIndexScalarNode>
GatherConstantIndexScalarNode::Clone() const {
  std::string name = GetName();
  return std::make_shared<GatherConstantIndexScalarNode>(
      std::move(name), GetInput(), GetOutput(), GetIndex(), GetAxis());
}

int64_t GatherConstantIndexScalarNode::GetAxis() const noexcept {
  return axis_;
}

int64_t GatherConstantIndexScalarNode::GetIndex() const noexcept {
  return index_;
}

GatherConstantDataTensorNode::GatherConstantDataTensorNode(
    std::string &&name, std::shared_ptr<Region> &&input,
    std::shared_ptr<Region> &&output, Tensor &&data, int64_t axis)
    : Node(std::move(name)), GatherNode(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      data_(data), axis_(axis) {
#ifdef DEBUG
  assert(input_ != nullptr);
  assert(output_ != nullptr);
  // Only the gather index of 0 is supported currently. If new cases occur, the
  // code should be updated.
  assert(axis == 0);
  const Meta &lhs_meta = data_.GetMeta();
  const Meta &rhs_meta = input_->GetMeta();
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

std::shared_ptr<SingleInputWithoutBufferNode>
GatherConstantDataTensorNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<GatherConstantDataTensorNode>
GatherConstantDataTensorNode::Clone() const {
  std::string name = GetName();
  Tensor data = GetData();
  return std::make_shared<GatherConstantDataTensorNode>(
      std::move(name), GetInput(), GetOutput(), std::move(data), GetAxis());
}

int64_t GatherConstantDataTensorNode::GetAxis() const noexcept { return axis_; }

const Tensor &GatherConstantDataTensorNode::GetData() const noexcept {
  return data_;
}

GatherConstantDataTensorAddTensorLhsAddTensorLhsNode::
    GatherConstantDataTensorAddTensorLhsAddTensorLhsNode(
        std::string &&name, Tensor &&data, Tensor &&add0_weight,
        Tensor &&add1_weight, std::shared_ptr<Region> &&input,
        std::shared_ptr<Region> &&output)
    : Node(std::move(name)), data_(std::move(data)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      add0_weight_(std::move(add0_weight)),
      add1_weight_(std::move(add1_weight)) {}

std::shared_ptr<SingleInputWithoutBufferNode>
GatherConstantDataTensorAddTensorLhsAddTensorLhsNode::
    CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<GatherConstantDataTensorAddTensorLhsAddTensorLhsNode>
GatherConstantDataTensorAddTensorLhsAddTensorLhsNode::Clone() const {
  std::string name = GetName();
  Tensor data = GetData(), add0_weight = GetAdd0Weight(),
         add1_weight = GetAdd1Weight();
  return std::make_shared<GatherConstantDataTensorAddTensorLhsAddTensorLhsNode>(
      std::move(name), std::move(data), std::move(add0_weight),
      std::move(add1_weight), GetInput(), GetOutput());
}

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
    std::string &&name, std::shared_ptr<Region> &&input,
    std::shared_ptr<Region> &&output, Tensor &&weights, Tensor &&bias,
    float64_t alpha = GemmNode::kAlpha, float64_t beta = GemmNode::kBeta,
    bool transA = GemmNode::kTransA, bool transB = GemmNode::kTransB)
    : Node(std::move(name)),
      GemmNode(std::move(name), alpha, beta, transA, transB),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      weights_(std::move(weights)), bias_(std::move(bias)) {
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

std::shared_ptr<SingleInputWithoutBufferNode>
GemmConstantWeightsBiasNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<GemmConstantWeightsBiasNode>
GemmConstantWeightsBiasNode::Clone() const {
  std::string name = GetName();
  Tensor weights = GetWeights(), bias = GetBias();
  return std::make_shared<GemmConstantWeightsBiasNode>(
      std::move(name), GetInput(), GetOutput(), std::move(weights),
      std::move(bias), GetAlpha(), GetBeta(), GetTransA(), GetTransB());
}

const Tensor &GemmConstantWeightsBiasNode::GetWeights() const noexcept {
  return weights_;
}

const Tensor &GemmConstantWeightsBiasNode::GetBias() const noexcept {
  return bias_;
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
    LayerNormalizationConstantScaleBiasNode(std::string &&name, Tensor &&scale,
                                            Tensor &&bias,
                                            std::shared_ptr<Region> &&input,
                                            std::shared_ptr<Region> &&output,
                                            int64_t axis, float64_t epsilon)
    : Node(std::move(name)),
      LayerNormalizationNode(std::move(name), axis, epsilon),
      SingleInputWithBufferNode(std::move(name), std::move(input),
                                std::move(output)),
      scale_(std::move(scale)), bias_(std::move(bias)) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == output_meta);
#endif
}

std::shared_ptr<SingleInputWithBufferNode>
LayerNormalizationConstantScaleBiasNode::CloneAsSingleInputWithBufferNode()
    const {
  return Clone();
}

std::shared_ptr<LayerNormalizationConstantScaleBiasNode>
LayerNormalizationConstantScaleBiasNode::Clone() const {
  std::string name = GetName();
  Tensor scale = GetScale(), bias = GetBias();
  return std::make_shared<LayerNormalizationConstantScaleBiasNode>(
      std::move(name), std::move(scale), std::move(bias), GetInput(),
      GetOutput(), GetAxis(), GetEpsilon());
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

const Tensor &
LayerNormalizationConstantScaleBiasNode::GetScale() const noexcept {
  return scale_;
}

const Tensor &
LayerNormalizationConstantScaleBiasNode::GetBias() const noexcept {
  return bias_;
}

MatMulNode::MatMulNode(std::string &&name) : Node(std::move(name)) {}

MatMulConstantLhsNode::MatMulConstantLhsNode(std::string &&name,
                                             Tensor &&weight,
                                             std::shared_ptr<Region> &&input,
                                             std::shared_ptr<Region> &&output)
    : Node(std::move(name)), MatMulNode(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      weight_(std::move(weight)) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  const std::vector<int64_t> &weight_shape = weight_.GetShape();
  const std::vector<int64_t> &input_shape = input_meta.GetShape();
  const std::vector<int64_t> &output_shape = output_meta.GetShape();
  size_t weight_shape_len = weight_shape.size();
  size_t input_shape_len = input_shape.size();
  size_t output_shape_len = output_shape.size();
  assert(weight_shape_len >= 2);
  assert(input_shape_len >= 2);
  assert(output_shape_len >= 2);
  size_t m = weight_shape[weight_shape_len - 2];
  size_t k = weight_shape[weight_shape_len - 1];
  size_t n = input_shape[input_shape_len - 1];
  assert(input_shape[input_shape_len - 2] == k);
  assert(output_shape[output_shape_len - 2] == m);
  assert(output_shape[output_shape_len - 1] == n);
  std::optional<Meta> expected_output_meta_opt = BroadcastMatMulShape(
      weight_.GetMeta(), input_meta, output_meta.GetType());
  assert(expected_output_meta_opt.has_value());
  assert(*expected_output_meta_opt == output_meta);
#endif
}

std::shared_ptr<SingleInputWithoutBufferNode>
MatMulConstantLhsNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<MatMulConstantLhsNode> MatMulConstantLhsNode::Clone() const {
  std::string name = GetName();
  Tensor weight = GetWeight();
  return std::make_shared<MatMulConstantLhsNode>(
      std::move(name), std::move(weight), GetInput(), GetOutput());
}

const Tensor &MatMulConstantLhsNode::GetWeight() const noexcept {
  return weight_;
}

MatMulConstantRhsNode::MatMulConstantRhsNode(std::string &&name,
                                             Tensor &&weight,
                                             std::shared_ptr<Region> &&input,
                                             std::shared_ptr<Region> &&output)
    : Node(std::move(name)), MatMulNode(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      weight_(std::move(weight)) {
#ifdef DEBUG
  const Meta &lhs_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  const std::vector<int64_t> &lhs_shape = lhs_meta.GetShape();
  const std::vector<int64_t> &rhs_shape = weight_.GetShape();
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
      BroadcastMatMulShape(lhs_meta, weight_.GetMeta(), output_meta.GetType());
  assert(expected_output_meta_opt.has_value());
  assert(*expected_output_meta_opt == output_meta);
#endif
}

std::shared_ptr<SingleInputWithoutBufferNode>
MatMulConstantRhsNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<MatMulConstantRhsNode> MatMulConstantRhsNode::Clone() const {
  std::string name = GetName();
  Tensor weight = GetWeight();
  return std::make_shared<MatMulConstantRhsNode>(
      std::move(name), std::move(weight), GetInput(), GetOutput());
}

const Tensor &MatMulConstantRhsNode::GetWeight() const noexcept {
  return weight_;
}

MatMulCommonNode::MatMulCommonNode(std::string &&name,
                                   std::shared_ptr<Region> &&lhs,
                                   std::shared_ptr<Region> &&rhs,
                                   std::shared_ptr<Region> &&output)
    : Node(std::move(name)), MatMulNode(std::move(name)),
      DoubleInputsWithoutBufferNode(std::move(name), std::move(lhs),
                                    std::move(rhs), std::move(output)) {
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

std::shared_ptr<DoubleInputsWithoutBufferNode>
MatMulCommonNode::CloneAsDoubleInputsWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<MatMulCommonNode> MatMulCommonNode::Clone() const {
  std::string name = GetName();
  return std::make_shared<MatMulCommonNode>(std::move(name), GetLhs(), GetRhs(),
                                            GetOutput());
}

MulNode::MulNode(std::string &&name) : Node(std::move(name)) {}

MulConstantNode::MulConstantNode(std::string &&name,
                                 std::shared_ptr<Region> &&input,
                                 std::shared_ptr<Region> &&output)
    : Node(std::move(name)), MulNode(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)) {}

std::shared_ptr<SingleInputWithoutBufferNode>
MulConstantNode::CloneAsSingleInputWithoutBufferNode() const {
  return CloneAsMulConstantNode();
}

MulConstantScalarNode::MulConstantScalarNode(std::string &&name,
                                             std::shared_ptr<Region> &&input,
                                             Type type, float64_t value,
                                             std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      MulConstantNode(std::move(name), std::move(input), std::move(output)),
      type_(type), value_(value) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == output_meta);
#endif
}

std::shared_ptr<MulConstantNode>
MulConstantScalarNode::CloneAsMulConstantNode() const {
  return Clone();
}

std::shared_ptr<MulConstantScalarNode> MulConstantScalarNode::Clone() const {
  std::string name = GetName();
  return std::make_shared<MulConstantScalarNode>(
      std::move(name), GetInput(), GetType(), GetValue(), GetOutput());
}

Type MulConstantScalarNode::GetType() const noexcept { return type_; }

float64_t MulConstantScalarNode::GetValue() const noexcept { return value_; }

MulConstantTensorNode::MulConstantTensorNode(std::string &&name,
                                             std::shared_ptr<Region> &&input,
                                             Tensor &&tensor,
                                             std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      MulConstantNode(std::move(name), std::move(input), std::move(output)),
      tensor_(std::move(tensor)) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == tensor_.GetMeta());
  assert(input_meta == output_meta);
#endif
}

std::shared_ptr<MulConstantNode>
MulConstantTensorNode::CloneAsMulConstantNode() const {
  return Clone();
}

std::shared_ptr<MulConstantTensorNode> MulConstantTensorNode::Clone() const {
  std::string name = GetName();
  Tensor tensor = GetTensor();
  return std::make_shared<MulConstantTensorNode>(
      std::move(name), GetInput(), std::move(tensor), GetOutput());
}

const Tensor &MulConstantTensorNode::GetTensor() const noexcept {
  return tensor_;
}

MulCommonNode::MulCommonNode(std::string &&name, std::shared_ptr<Region> &&lhs,
                             std::shared_ptr<Region> &&rhs,
                             std::shared_ptr<Region> &&output)
    : Node(std::move(name)), MulNode(std::move(name)),
      DoubleInputsWithoutBufferNode(std::move(name), std::move(lhs),
                                    std::move(rhs), std::move(output)) {}

std::shared_ptr<DoubleInputsWithoutBufferNode>
MulCommonNode::CloneAsDoubleInputsWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<MulCommonNode> MulCommonNode::Clone() const {
  std::string name = GetName();
  return std::make_shared<MulCommonNode>(std::move(name), GetLhs(), GetRhs(),
                                         GetOutput());
}

PowNode::PowNode(std::string &&name, Type type, float64_t exp,
                 std::shared_ptr<Region> &&input,
                 std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      type_(type), exp_(exp) {}

std::shared_ptr<SingleInputWithoutBufferNode>
PowNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<PowNode> PowNode::Clone() const {
  std::string name = GetName();
  return std::make_shared<PowNode>(std::move(name), GetType(), GetExp(),
                                   GetInput(), GetOutput());
}

Type PowNode::GetType() const noexcept { return type_; }

float64_t PowNode::GetExp() const noexcept { return exp_; }

ReshapeNode::ReshapeNode(std::string &&name, std::shared_ptr<Region> &&input,
                         std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)) {
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

std::shared_ptr<SingleInputWithoutBufferNode>
ReshapeNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<ReshapeNode> ReshapeNode::Clone() const {
  std::string name = GetName();
  return std::make_shared<ReshapeNode>(std::move(name), GetInput(),
                                       GetOutput());
}

SoftmaxNode::SoftmaxNode(std::string &&name, std::shared_ptr<Region> &&input,
                         std::shared_ptr<Region> &&output, int64_t axis)
    : Node(std::move(name)),
      SingleInputWithBufferNode(std::move(name), std::move(input),
                                std::move(output)),
      axis_(axis) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == output_meta);
#endif
}

std::shared_ptr<SingleInputWithBufferNode>
SoftmaxNode::CloneAsSingleInputWithBufferNode() const {
  return Clone();
}

std::shared_ptr<SoftmaxNode> SoftmaxNode::Clone() const {
  std::string name = GetName();
  return std::make_shared<SoftmaxNode>(std::move(name), GetInput(), GetOutput(),
                                       GetAxis());
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

SplitNode::SplitNode(std::string &&name, std::shared_ptr<Region> &&input,
                     std::vector<std::shared_ptr<Region>> &&outputs,
                     int64_t axis)
    : Node(std::move(name)), input_(std::move(input)),
      outputs_(std::move(outputs)), axis_(axis) {
#ifdef DEBUG
  const Meta &input_meta = input_->GetMeta();
  const std::vector<int64_t> &input_shape = input_meta.GetShape();
  size_t input_shape_len = input_shape.size();
  assert(axis < input_shape_len);
  size_t sum = 0;
  for (const std::shared_ptr<Region> &output : outputs_) {
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

std::shared_ptr<Node> SplitNode::CloneAsNode() const { return Clone(); }

std::shared_ptr<SplitNode> SplitNode::Clone() const {
  std::string name = GetName();
  std::vector<std::shared_ptr<Region>> outputs = GetOutputs();
  return std::make_shared<SplitNode>(std::move(name), GetInput(),
                                     std::move(outputs), GetAxis());
}

std::shared_ptr<Region> SplitNode::GetInput() const noexcept { return input_; }

const std::vector<std::shared_ptr<Region>> &
SplitNode::GetOutputs() const noexcept {
  return outputs_;
}

int64_t SplitNode::GetAxis() const noexcept {
  return axis_ >= 0 ? axis_ : GetMeta().GetShape().size() + axis_;
}

const Meta &SplitNode::GetMeta() const noexcept { return input_->GetMeta(); }

SubNode::SubNode(std::string &&name) : Node(std::move(name)) {}

SubConstantScalarLhsNode::SubConstantScalarLhsNode(
    std::string &&name, Type type, float64_t value,
    std::shared_ptr<Region> &&input, std::shared_ptr<Region> &&output)
    : Node(std::move(name)), SubNode(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      type_(type), value_(value) {}

std::shared_ptr<SingleInputWithoutBufferNode>
SubConstantScalarLhsNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<SubConstantScalarLhsNode>
SubConstantScalarLhsNode::Clone() const {
  std::string name = GetName();
  return std::make_shared<SubConstantScalarLhsNode>(
      std::move(name), GetType(), GetValue(), GetInput(), GetOutput());
}

Type SubConstantScalarLhsNode::GetType() const noexcept { return type_; }

float64_t SubConstantScalarLhsNode::GetValue() const noexcept { return value_; }

TanhNode::TanhNode(std::string &&name, std::shared_ptr<Region> &&input,
                   std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)) {}

std::shared_ptr<SingleInputWithoutBufferNode>
TanhNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<TanhNode> TanhNode::Clone() const {
  std::string name = GetName();
  return std::make_shared<TanhNode>(std::move(name), GetInput(), GetOutput());
}

TransposeNode::TransposeNode(std::string &&name, std::vector<int64_t> &&perm,
                             std::shared_ptr<Region> &&input,
                             std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      perm_(std::move(perm)) {
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

std::shared_ptr<SingleInputWithoutBufferNode>
TransposeNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<TransposeNode> TransposeNode::Clone() const {
  std::string name = GetName();
  std::vector<int64_t> perm = GetPerm();
  return std::make_shared<TransposeNode>(std::move(name), std::move(perm),
                                         GetInput(), GetOutput());
}

const std::vector<int64_t> &TransposeNode::GetPerm() const noexcept {
  return perm_;
}

UnsqueezeNode::UnsqueezeNode(std::string &&name, std::vector<int64_t> &&axes,
                             std::shared_ptr<Region> &&input,
                             std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      axes_(std::move(axes)) {
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

std::shared_ptr<SingleInputWithoutBufferNode>
UnsqueezeNode::CloneAsSingleInputWithoutBufferNode() const {
  return Clone();
}

std::shared_ptr<UnsqueezeNode> UnsqueezeNode::Clone() const {
  std::string name = GetName();
  std::vector<int64_t> axes = GetAxes();
  return std::make_shared<UnsqueezeNode>(std::move(name), std::move(axes),
                                         GetInput(), GetOutput());
}

const std::vector<int64_t> &UnsqueezeNode::GetAxes() const noexcept {
  return axes_;
}

UnsqueezeSubLhsScalarMulRhsScalarNode::UnsqueezeSubLhsScalarMulRhsScalarNode(
    std::string &&name, std::vector<int64_t> &&unsqueeze_axes, Type sub_type,
    float64_t sub_val, Type mul_type, float64_t mul_val,
    std::shared_ptr<Region> &&input, std::shared_ptr<Region> &&output)
    : Node(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      unsqueeze_axes_(std::move(unsqueeze_axes)), sub_type_(sub_type),
      sub_val_(sub_val), mul_type_(mul_type), mul_val_(mul_val) {
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

std::shared_ptr<SingleInputWithoutBufferNode>
UnsqueezeSubLhsScalarMulRhsScalarNode::CloneAsSingleInputWithoutBufferNode()
    const {
  return Clone();
}

std::shared_ptr<UnsqueezeSubLhsScalarMulRhsScalarNode>
UnsqueezeSubLhsScalarMulRhsScalarNode::Clone() const {
  std::string name = GetName();
  std::vector<int64_t> unsqueeze_axes = GetUnsqueezeAxes();
  return std::make_shared<UnsqueezeSubLhsScalarMulRhsScalarNode>(
      std::move(name), std::move(unsqueeze_axes), GetSubType(), GetSubVal(),
      GetMulType(), GetMulVal(), GetInput(), GetOutput());
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
    std::string &&name, Tensor &&cond, Type type, float64_t y,
    std::shared_ptr<Region> &&input, std::shared_ptr<Region> &&output)
    : Node(std::move(name)), WhereNode(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      cond_(std::move(cond)), type_(type), y_(y) {
#ifdef DEBUG
  assert(cond_.GetType() == Type::BOOL);
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  std::optional<Meta> broadcasted_meta_opt =
      BroadcastShape(cond_.GetMeta(), input_meta, output_meta.GetType());
  assert(broadcasted_meta_opt.has_value());
  assert(*broadcasted_meta_opt == output_meta);
#endif
}

std::shared_ptr<SingleInputWithoutBufferNode>
WhereConstantCondConstantScalarYNode::CloneAsSingleInputWithoutBufferNode()
    const {
  return Clone();
}

std::shared_ptr<WhereConstantCondConstantScalarYNode>
WhereConstantCondConstantScalarYNode::Clone() const {
  std::string name = GetName();
  Tensor cond = GetCond();
  return std::make_shared<WhereConstantCondConstantScalarYNode>(
      std::move(name), std::move(cond), GetType(), GetY(), GetInput(),
      GetOutput());
}

const Tensor &WhereConstantCondConstantScalarYNode::GetCond() const noexcept {
  return cond_;
}

Type WhereConstantCondConstantScalarYNode::GetType() const noexcept {
  return type_;
}

float64_t WhereConstantCondConstantScalarYNode::GetY() const noexcept {
  return y_;
}

WhereConstantCondConstantTensorYNode::WhereConstantCondConstantTensorYNode(
    std::string &&name, Tensor &&cond, Tensor &&y,
    std::shared_ptr<Region> &&input, std::shared_ptr<Region> &&output)
    : Node(std::move(name)), WhereNode(std::move(name)),
      SingleInputWithoutBufferNode(std::move(name), std::move(input),
                                   std::move(output)),
      cond_(std::move(cond)), y_(std::move(y)) {
#ifdef DEBUG
  assert(cond_.GetType() == Type::BOOL);
  const Meta &input_meta = input_->GetMeta();
  const Meta &output_meta = output_->GetMeta();
  assert(input_meta == cond_.GetMeta());
  assert(input_meta == y_.GetMeta());
  assert(input_meta == output_meta);
#endif
}

std::shared_ptr<SingleInputWithoutBufferNode>
WhereConstantCondConstantTensorYNode::CloneAsSingleInputWithoutBufferNode()
    const {
  return Clone();
}

std::shared_ptr<WhereConstantCondConstantTensorYNode>
WhereConstantCondConstantTensorYNode::Clone() const {
  std::string name = GetName();
  Tensor cond = GetCond(), y = GetY();
  return std::make_shared<WhereConstantCondConstantTensorYNode>(
      std::move(name), std::move(cond), std::move(y), GetInput(), GetOutput());
}

const Tensor &WhereConstantCondConstantTensorYNode::GetCond() const noexcept {
  return cond_;
}

const Tensor &WhereConstantCondConstantTensorYNode::GetY() const noexcept {
  return y_;
}

} // namespace flow
} // namespace cpu_transformers
