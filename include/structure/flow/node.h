#ifndef CPU_TRANSFORMERS_STRUCTURE_FLOW_NODE_H_
#define CPU_TRANSFORMERS_STRUCTURE_FLOW_NODE_H_

#include "structure/flow/fwd.h"
#include "structure/tensor/tensor.h"
#include "utils/float.h"
#include "utils/type.h"
#include <cstdint>
#include <memory>
#include <string>

// TODO: We only implements part of the cases in the ONNX. Our current target is
// to make it work on Bert and GPT-2. If new cases occur, it will throw an error
// in the DEBUG mode, and we should consider implementing it.

namespace cpu_transformers {
namespace flow {

class Node {
public:
  Node(std::string &&name);
  Node(const Node &node) = delete;
  Node(Node &&node) = default;
  virtual ~Node() = default;
  virtual std::shared_ptr<Node> CloneAsNode() const = 0;
  virtual const std::string &GetName() const noexcept;
  virtual size_t GetBufferSize() const noexcept;

protected:
  const std::string name_;
};

class SingleInputNode : virtual public Node {
public:
  SingleInputNode(std::string &&name, std::shared_ptr<Region> &&input,
                  std::shared_ptr<Region> &&output);
  SingleInputNode(const SingleInputNode &node) = delete;
  SingleInputNode(SingleInputNode &&node) = default;
  virtual ~SingleInputNode() = default;
  virtual std::shared_ptr<Node> CloneAsNode() const override;
  virtual std::shared_ptr<SingleInputNode> CloneAsSingleInputNode() const = 0;
  std::shared_ptr<Region> GetInput() const noexcept;
  std::shared_ptr<Region> GetOutput() const noexcept;
  const std::string &GetInputAsString() const noexcept;
  const std::string &GetOutputAsString() const noexcept;
  void SetInput(std::shared_ptr<Region> &&input) noexcept;
  void SetOutput(std::shared_ptr<Region> &&output) noexcept;

protected:
  std::shared_ptr<Region> input_;
  std::shared_ptr<Region> output_;
};

class DoubleInputsNode : virtual public Node {
public:
  DoubleInputsNode(std::string &&name, std::shared_ptr<Region> &&lhs,
                   std::shared_ptr<Region> &&rhs,
                   std::shared_ptr<Region> &&output);
  DoubleInputsNode(const DoubleInputsNode &node) = delete;
  DoubleInputsNode(DoubleInputsNode &&node) = default;
  virtual ~DoubleInputsNode() = default;
  virtual std::shared_ptr<Node> CloneAsNode() const override;
  virtual std::shared_ptr<DoubleInputsNode> CloneAsDoubleInputsNode() const = 0;
  std::shared_ptr<Region> GetLhs() const noexcept;
  std::shared_ptr<Region> GetRhs() const noexcept;
  std::shared_ptr<Region> GetOutput() const noexcept;
  const std::string &GetLhsAsString() const noexcept;
  const std::string &GetRhsAsString() const noexcept;
  const std::string &GetOutputAsString() const noexcept;
  void SetLhs(std::shared_ptr<Region> &&lhs) noexcept;
  void SetRhs(std::shared_ptr<Region> &&rhs) noexcept;
  void SetOutput(std::shared_ptr<Region> &&output) noexcept;

protected:
  std::shared_ptr<Region> lhs_;
  std::shared_ptr<Region> rhs_;
  std::shared_ptr<Region> output_;
};

class SingleInputWithoutBufferNode : public SingleInputNode {
public:
  SingleInputWithoutBufferNode(std::string &&name,
                               std::shared_ptr<Region> &&input,
                               std::shared_ptr<Region> &&output);
  SingleInputWithoutBufferNode(const SingleInputWithoutBufferNode &node) =
      delete;
  SingleInputWithoutBufferNode(SingleInputWithoutBufferNode &&node) = default;
  virtual ~SingleInputWithoutBufferNode() = default;
  virtual std::shared_ptr<SingleInputNode>
  CloneAsSingleInputNode() const override;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const = 0;
};

class SingleInputWithBufferNode : public SingleInputNode {
public:
  SingleInputWithBufferNode(std::string &&name, std::shared_ptr<Region> &&input,
                            std::shared_ptr<Region> &&output);
  SingleInputWithBufferNode(const SingleInputWithBufferNode &node) = delete;
  SingleInputWithBufferNode(SingleInputWithBufferNode &&node) = default;
  virtual ~SingleInputWithBufferNode() = default;
  virtual std::shared_ptr<SingleInputNode>
  CloneAsSingleInputNode() const override;
  virtual std::shared_ptr<SingleInputWithBufferNode>
  CloneAsSingleInputWithBufferNode() const = 0;
};

class DoubleInputsWithoutBufferNode : public DoubleInputsNode {
public:
  DoubleInputsWithoutBufferNode(std::string &&name,
                                std::shared_ptr<Region> &&lhs,
                                std::shared_ptr<Region> &&rhs,
                                std::shared_ptr<Region> &&output);
  DoubleInputsWithoutBufferNode(const DoubleInputsWithoutBufferNode &node) =
      delete;
  DoubleInputsWithoutBufferNode(DoubleInputsWithoutBufferNode &&node) = default;
  virtual ~DoubleInputsWithoutBufferNode() = default;
  virtual std::shared_ptr<DoubleInputsNode>
  CloneAsDoubleInputsNode() const override;
  virtual std::shared_ptr<DoubleInputsWithoutBufferNode>
  CloneAsDoubleInputsWithoutBufferNode() const = 0;
};

class DoubleInputsWithBufferNode : virtual public DoubleInputsNode {
public:
  DoubleInputsWithBufferNode(std::string &&name, std::shared_ptr<Region> &&lhs,
                             std::shared_ptr<Region> &&rhs,
                             std::shared_ptr<Region> &&output);
  DoubleInputsWithBufferNode(const DoubleInputsWithBufferNode &node) = delete;
  DoubleInputsWithBufferNode(DoubleInputsWithBufferNode &&node) = default;
  virtual ~DoubleInputsWithBufferNode() = default;
  virtual std::shared_ptr<DoubleInputsNode>
  CloneAsDoubleInputsNode() const override;
  virtual std::shared_ptr<DoubleInputsWithBufferNode>
  CloneAsDoubleInputsWithBufferNode() const = 0;
};

class AddNode : virtual public Node {
public:
  AddNode(std::string &&name);
  AddNode(const AddNode &node) = delete;
  AddNode(AddNode &&node) = default;
  virtual ~AddNode() = default;
};

class AddConstantNode : public AddNode, public SingleInputWithoutBufferNode {
public:
  AddConstantNode(std::string &&name, Type type, float64_t value,
                  std::shared_ptr<Region> &&input,
                  std::shared_ptr<Region> &&output);
  AddConstantNode(const AddConstantNode &node) = delete;
  AddConstantNode(AddConstantNode &&node) = default;
  virtual ~AddConstantNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<AddConstantNode> Clone() const;
  Type GetType() const noexcept;
  float64_t GetValue() const noexcept;

private:
  const Type type_;
  const float64_t value_;
};

class AddCommonNode : public AddNode, public DoubleInputsWithoutBufferNode {
public:
  AddCommonNode(std::string &&name, std::shared_ptr<Region> &&lhs,
                std::shared_ptr<Region> &&rhs,
                std::shared_ptr<Region> &&output);
  AddCommonNode(const AddCommonNode &node) = delete;
  AddCommonNode(AddCommonNode &&node) = default;
  virtual ~AddCommonNode() = default;
  virtual std::shared_ptr<DoubleInputsWithoutBufferNode>
  CloneAsDoubleInputsWithoutBufferNode() const override;
  std::shared_ptr<AddCommonNode> Clone() const;
};

class AddDivErfAddMulMulNode : public SingleInputWithoutBufferNode {
public:
  AddDivErfAddMulMulNode(std::string &&name, Tensor &&add0_weight,
                         Type div_type, float64_t div_weight, Type add1_type,
                         float64_t add1_weight, Type mul1_type,
                         float64_t mul1_weight, std::shared_ptr<Region> &&input,
                         std::shared_ptr<Region> &&output);
  AddDivErfAddMulMulNode(const AddDivErfAddMulMulNode &node) = delete;
  AddDivErfAddMulMulNode(AddDivErfAddMulMulNode &&node) = default;
  virtual ~AddDivErfAddMulMulNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<AddDivErfAddMulMulNode> Clone() const;
  const Tensor &GetAdd0Weight() const noexcept;
  Type GetDivType() const noexcept;
  float64_t GetDivWeight() const noexcept;
  Type GetAdd1Type() const noexcept;
  float64_t GetAdd1Weight() const noexcept;
  Type GetMul1Type() const noexcept;
  float64_t GetMul1Weight() const noexcept;

private:
  const Tensor add0_weight_;
  const Type div_type_;
  const float64_t div_weight_;
  const Type add1_type_;
  const float64_t add1_weight_;
  const Type mul1_type_;
  const float64_t mul1_weight_;
};

class DivNode : virtual public Node {
public:
  DivNode(std::string &&name);
  DivNode(const DivNode &node) = delete;
  DivNode(DivNode &&node) = default;
  virtual ~DivNode() = default;
};

class DivConstantScalarNode : public DivNode,
                              public SingleInputWithoutBufferNode {
public:
  DivConstantScalarNode(std::string &&name, Type type, float64_t value,
                        std::shared_ptr<Region> &&input,
                        std::shared_ptr<Region> &&output);
  DivConstantScalarNode(const DivConstantScalarNode &node) = delete;
  DivConstantScalarNode(DivConstantScalarNode &&node) = default;
  virtual ~DivConstantScalarNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<DivConstantScalarNode> Clone() const;
  Type GetType() const noexcept;
  float64_t GetValue() const noexcept;

private:
  const Type type_;
  const float64_t value_;
};

class ErfNode : public SingleInputWithoutBufferNode {
public:
  ErfNode(std::string &&name, std::shared_ptr<Region> &&input,
          std::shared_ptr<Region> &&output);
  ErfNode(const ErfNode &node) = delete;
  ErfNode(ErfNode &&node) = default;
  virtual ~ErfNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<ErfNode> Clone() const;
};

class GatherNode : virtual public Node {
public:
  constexpr static int64_t kAxis = 0;
  constexpr static const char *kAxisAttrName = "axis";
  GatherNode(std::string &&name);
  GatherNode(const GatherNode &node) = delete;
  GatherNode(GatherNode &&node) = default;
  virtual ~GatherNode() = default;
};

class GatherConstantIndexScalarNode : public GatherNode,
                                      public SingleInputWithoutBufferNode {
public:
  GatherConstantIndexScalarNode(std::string &&name,
                                std::shared_ptr<Region> &&input,
                                std::shared_ptr<Region> &&output, int64_t index,
                                int64_t axis = kAxis);
  GatherConstantIndexScalarNode(const GatherConstantIndexScalarNode &node) =
      delete;
  GatherConstantIndexScalarNode(GatherConstantIndexScalarNode &&node) = default;
  virtual ~GatherConstantIndexScalarNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<GatherConstantIndexScalarNode> Clone() const;
  int64_t GetAxis() const noexcept;
  int64_t GetIndex() const noexcept;

private:
  const int64_t axis_;
  const int64_t index_;
};

class GatherConstantDataTensorNode : public GatherNode,
                                     public SingleInputWithoutBufferNode {
public:
  GatherConstantDataTensorNode(std::string &&name,
                               std::shared_ptr<Region> &&rhs,
                               std::shared_ptr<Region> &&output, Tensor &&data,
                               int64_t axis = kAxis);
  GatherConstantDataTensorNode(const GatherConstantDataTensorNode &node) =
      delete;
  GatherConstantDataTensorNode(GatherConstantDataTensorNode &&node) = default;
  virtual ~GatherConstantDataTensorNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<GatherConstantDataTensorNode> Clone() const;
  int64_t GetAxis() const noexcept;
  const Tensor &GetData() const noexcept;

private:
  const Tensor data_;
  const int64_t axis_;
};

class GatherConstantDataTensorAddTensorLhsAddTensorLhsNode
    : public SingleInputWithoutBufferNode {
public:
  GatherConstantDataTensorAddTensorLhsAddTensorLhsNode(
      std::string &&name, Tensor &&data, Tensor &&add0_weight,
      Tensor &&add1_weight, std::shared_ptr<Region> &&input,
      std::shared_ptr<Region> &&output);
  GatherConstantDataTensorAddTensorLhsAddTensorLhsNode(
      const GatherConstantDataTensorAddTensorLhsAddTensorLhsNode &node) =
      delete;
  GatherConstantDataTensorAddTensorLhsAddTensorLhsNode(
      GatherConstantDataTensorAddTensorLhsAddTensorLhsNode &&node) = default;
  virtual ~GatherConstantDataTensorAddTensorLhsAddTensorLhsNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<GatherConstantDataTensorAddTensorLhsAddTensorLhsNode>
  Clone() const;
  const Tensor &GetData() const noexcept;
  const Tensor &GetAdd0Weight() const noexcept;
  const Tensor &GetAdd1Weight() const noexcept;

private:
  const Tensor data_;
  const Tensor add0_weight_;
  const Tensor add1_weight_;
};

class GemmNode : virtual public Node {
public:
  GemmNode(std::string &&name, float64_t alpha, float64_t beta, bool transA,
           bool transB);
  GemmNode(const GemmNode &node) = delete;
  GemmNode(GemmNode &&node) = default;
  virtual ~GemmNode() = default;
  static constexpr float64_t kAlpha = 1.0;
  static constexpr float64_t kBeta = 1.0;
  static constexpr bool kTransA = false;
  static constexpr bool kTransB = false;
  static constexpr const char *kAlphaAttrName = "alpha";
  static constexpr const char *kBetaAttrName = "beta";
  static constexpr const char *kTransAAttrName = "transA";
  static constexpr const char *kTransBAttrName = "transB";
  float64_t GetAlpha() const noexcept;
  float64_t GetBeta() const noexcept;
  bool GetTransA() const noexcept;
  bool GetTransB() const noexcept;

protected:
  const float64_t alpha_;
  const float64_t beta_;
  const bool transA_;
  const bool transB_;
};

class GemmConstantWeightsBiasNode : public GemmNode,
                                    public SingleInputWithoutBufferNode {
public:
  GemmConstantWeightsBiasNode(std::string &&name,
                              std::shared_ptr<Region> &&input,
                              std::shared_ptr<Region> &&output,
                              Tensor &&weights, Tensor &&bias, float64_t alpha,
                              float64_t beta, bool transA, bool transB);
  GemmConstantWeightsBiasNode(const GemmConstantWeightsBiasNode &node) = delete;
  GemmConstantWeightsBiasNode(GemmConstantWeightsBiasNode &&node) = default;
  virtual ~GemmConstantWeightsBiasNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<GemmConstantWeightsBiasNode> Clone() const;
  const Tensor &GetWeights() const noexcept;
  const Tensor &GetBias() const noexcept;

private:
  const Tensor weights_;
  const Tensor bias_;
};

class LayerNormalizationNode : virtual public Node {
public:
  LayerNormalizationNode(std::string &&name, int64_t axis, float64_t epsilon);
  LayerNormalizationNode(const LayerNormalizationNode &node) = delete;
  LayerNormalizationNode(LayerNormalizationNode &&node) = default;
  virtual ~LayerNormalizationNode() = default;
  static constexpr int64_t kAxis = 1;
  static constexpr float64_t kEpsilon = 1e-5;
  static constexpr const char *kAxisAttrName = "axis";
  static constexpr const char *kEpsilonAttrName = "epsilon";
  virtual const Meta &GetMeta() const noexcept = 0;
  int64_t GetAxis() const noexcept;
  float64_t GetEpsilon() const noexcept;

protected:
  const int64_t axis_;
  const float64_t epsilon_;
};

class LayerNormalizationConstantScaleBiasNode
    : public LayerNormalizationNode,
      public SingleInputWithBufferNode {
public:
  LayerNormalizationConstantScaleBiasNode(std::string &&name, Tensor &&scale,
                                          Tensor &&bias,
                                          std::shared_ptr<Region> &&input,
                                          std::shared_ptr<Region> &&output,
                                          int64_t axis, float64_t epsilon);
  LayerNormalizationConstantScaleBiasNode(
      const LayerNormalizationConstantScaleBiasNode &node) = delete;
  LayerNormalizationConstantScaleBiasNode(
      LayerNormalizationConstantScaleBiasNode &&node) = default;
  virtual ~LayerNormalizationConstantScaleBiasNode() = default;
  virtual std::shared_ptr<SingleInputWithBufferNode>
  CloneAsSingleInputWithBufferNode() const override;
  std::shared_ptr<LayerNormalizationConstantScaleBiasNode> Clone() const;
  size_t GetBufferSize() const noexcept override;
  const Meta &GetMeta() const noexcept override;
  const Tensor &GetScale() const noexcept;
  const Tensor &GetBias() const noexcept;

private:
  const Tensor scale_;
  const Tensor bias_;
};

class MatMulNode : public DoubleInputsWithoutBufferNode {
public:
  MatMulNode(std::string &&name, std::shared_ptr<Region> &&lhs,
             std::shared_ptr<Region> &&rhs, std::shared_ptr<Region> &&output);
  MatMulNode(const MatMulNode &node) = delete;
  MatMulNode(MatMulNode &&node) = default;
  virtual ~MatMulNode() = default;
  virtual std::shared_ptr<DoubleInputsWithoutBufferNode>
  CloneAsDoubleInputsWithoutBufferNode() const override;
  std::shared_ptr<MatMulNode> Clone() const;
};

class MulNode : virtual public Node {
public:
  MulNode(std::string &&name);
  MulNode(const MulNode &node) = delete;
  MulNode(MulNode &&node) = default;
  virtual ~MulNode() = default;
};

class MulConstantNode : public MulNode, public SingleInputWithoutBufferNode {
public:
  MulConstantNode(std::string &&name, std::shared_ptr<Region> &&input,
                  Type type, float64_t value, std::shared_ptr<Region> &&output);
  MulConstantNode(const MulConstantNode &node) = delete;
  MulConstantNode(MulConstantNode &&node) = default;
  virtual ~MulConstantNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<MulConstantNode> Clone() const;
  Type GetType() const noexcept;
  float64_t GetValue() const noexcept;

private:
  const Type type_;
  const float64_t value_;
};

class MulCommonNode : public MulNode, public DoubleInputsWithoutBufferNode {
public:
  MulCommonNode(std::string &&name, std::shared_ptr<Region> &&lhs,
                std::shared_ptr<Region> &&rhs,
                std::shared_ptr<Region> &&output);
  MulCommonNode(const MulCommonNode &node) = delete;
  MulCommonNode(MulCommonNode &&node) = default;
  virtual ~MulCommonNode() = default;
  virtual std::shared_ptr<DoubleInputsWithoutBufferNode>
  CloneAsDoubleInputsWithoutBufferNode() const override;
  std::shared_ptr<MulCommonNode> Clone() const;
};

class PowNode : public SingleInputWithoutBufferNode {
public:
  PowNode(std::string &&name, Type type, float64_t exp,
          std::shared_ptr<Region> &&input, std::shared_ptr<Region> &&output);
  PowNode(const PowNode &node) = delete;
  PowNode(PowNode &&node) = default;
  virtual ~PowNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<PowNode> Clone() const;
  Type GetType() const noexcept;
  float64_t GetExp() const noexcept;

private:
  const Type type_;
  const float64_t exp_;
};

class ReshapeNode : public SingleInputWithoutBufferNode {
public:
  ReshapeNode(std::string &&name, std::shared_ptr<Region> &&input,
              std::shared_ptr<Region> &&output);
  ReshapeNode(const ReshapeNode &node) = delete;
  ReshapeNode(ReshapeNode &&node) = default;
  virtual ~ReshapeNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<ReshapeNode> Clone() const;
};

class SoftmaxNode : public SingleInputWithBufferNode {
public:
  constexpr static int64_t kAxis = -1;
  constexpr static const char *kAxisAttrName = "axis";
  SoftmaxNode(std::string &&name, std::shared_ptr<Region> &&input,
              std::shared_ptr<Region> &&output, int64_t axis = kAxis);
  SoftmaxNode(const SoftmaxNode &node) = delete;
  SoftmaxNode(SoftmaxNode &&node) = default;
  virtual ~SoftmaxNode() = default;
  virtual std::shared_ptr<SingleInputWithBufferNode>
  CloneAsSingleInputWithBufferNode() const override;
  std::shared_ptr<SoftmaxNode> Clone() const;
  size_t GetBufferSize() const noexcept override;
  int64_t GetAxis() const noexcept;
  const Meta &GetMeta() const noexcept;

private:
  const int64_t axis_;
};

class SubNode : virtual public Node {
public:
  SubNode(std::string &&name);
  SubNode(const SubNode &node) = delete;
  SubNode(SubNode &&node) = default;
  virtual ~SubNode() = default;
};

class SubConstantScalarLhsNode : public SubNode,
                                 public SingleInputWithoutBufferNode {
public:
  SubConstantScalarLhsNode(std::string &&name, Type type, float64_t value,
                           std::shared_ptr<Region> &&input,
                           std::shared_ptr<Region> &&output);
  SubConstantScalarLhsNode(const SubConstantScalarLhsNode &node) = delete;
  SubConstantScalarLhsNode(SubConstantScalarLhsNode &&node) = default;
  virtual ~SubConstantScalarLhsNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<SubConstantScalarLhsNode> Clone() const;
  Type GetType() const noexcept;
  float64_t GetValue() const noexcept;

private:
  const Type type_;
  const float64_t value_;
};

class TanhNode : public SingleInputWithoutBufferNode {
public:
  TanhNode(std::string &&name, std::shared_ptr<Region> &&input,
           std::shared_ptr<Region> &&output);
  TanhNode(const TanhNode &node) = delete;
  TanhNode(TanhNode &&node) = default;
  virtual ~TanhNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<TanhNode> Clone() const;
};

class TransposeNode : public SingleInputWithoutBufferNode {
public:
  constexpr static const char *kPermAttrName = "perm";
  TransposeNode(std::string &&name, std::vector<int64_t> &&perm,
                std::shared_ptr<Region> &&input,
                std::shared_ptr<Region> &&output);
  TransposeNode(const TransposeNode &node) = delete;
  TransposeNode(TransposeNode &&node) = default;
  virtual ~TransposeNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<TransposeNode> Clone() const;
  const std::vector<int64_t> &GetPerm() const noexcept;

private:
  const std::vector<int64_t> perm_;
};

class UnsqueezeNode : public SingleInputWithoutBufferNode {
public:
  UnsqueezeNode(std::string &&name, std::vector<int64_t> &&axes,
                std::shared_ptr<Region> &&input,
                std::shared_ptr<Region> &&output);
  UnsqueezeNode(const UnsqueezeNode &node) = delete;
  UnsqueezeNode(UnsqueezeNode &&node) = default;
  virtual ~UnsqueezeNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<UnsqueezeNode> Clone() const;
  const std::vector<int64_t> &GetAxes() const noexcept;

private:
  const std::vector<int64_t> axes_;
};

class UnsqueezeSubLhsScalarMulRhsScalarNode
    : public SingleInputWithoutBufferNode {
public:
  UnsqueezeSubLhsScalarMulRhsScalarNode(std::string &&name,
                                        std::vector<int64_t> &&unsqueeze_axes,
                                        Type sub_type, float64_t sub_val,
                                        Type mul_type, float64_t mul_val,
                                        std::shared_ptr<Region> &&input,
                                        std::shared_ptr<Region> &&output);
  UnsqueezeSubLhsScalarMulRhsScalarNode(
      const UnsqueezeSubLhsScalarMulRhsScalarNode &node) = delete;
  UnsqueezeSubLhsScalarMulRhsScalarNode(
      UnsqueezeSubLhsScalarMulRhsScalarNode &&node) = default;
  virtual ~UnsqueezeSubLhsScalarMulRhsScalarNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<UnsqueezeSubLhsScalarMulRhsScalarNode> Clone() const;
  const std::vector<int64_t> &GetUnsqueezeAxes() const noexcept;
  Type GetSubType() const noexcept;
  float64_t GetSubVal() const noexcept;
  Type GetMulType() const noexcept;
  float64_t GetMulVal() const noexcept;

private:
  std::vector<int64_t> unsqueeze_axes_;
  const Type sub_type_;
  const float64_t sub_val_;
  const Type mul_type_;
  const float64_t mul_val_;
};

class WhereNode : virtual public Node {
public:
  WhereNode(std::string &&name);
  WhereNode(const WhereNode &node) = delete;
  WhereNode(WhereNode &&node) = default;
  virtual ~WhereNode() = default;
};

class WhereConstantCondConstantScalarYNode
    : public WhereNode,
      public SingleInputWithoutBufferNode {
public:
  WhereConstantCondConstantScalarYNode(std::string &&name, Tensor &&cond,
                                       Type type, float64_t y,
                                       std::shared_ptr<Region> &&input,
                                       std::shared_ptr<Region> &&output);
  WhereConstantCondConstantScalarYNode(
      const WhereConstantCondConstantScalarYNode &node) = delete;
  WhereConstantCondConstantScalarYNode(
      WhereConstantCondConstantScalarYNode &&node) = default;
  virtual ~WhereConstantCondConstantScalarYNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<WhereConstantCondConstantScalarYNode> Clone() const;
  const Tensor &GetCond() const noexcept;
  Type GetType() const noexcept;
  float64_t GetY() const noexcept;

private:
  const Tensor cond_;
  const Type type_;
  const float64_t y_;
};

class WhereConstantCondConstantTensorYNode
    : public WhereNode,
      public SingleInputWithoutBufferNode {
public:
  WhereConstantCondConstantTensorYNode(std::string &&name, Tensor &&cond,
                                       Tensor &&y,
                                       std::shared_ptr<Region> &&input,
                                       std::shared_ptr<Region> &&output);
  WhereConstantCondConstantTensorYNode(
      const WhereConstantCondConstantTensorYNode &node) = delete;
  WhereConstantCondConstantTensorYNode(
      WhereConstantCondConstantTensorYNode &&node) = default;
  virtual ~WhereConstantCondConstantTensorYNode() = default;
  virtual std::shared_ptr<SingleInputWithoutBufferNode>
  CloneAsSingleInputWithoutBufferNode() const override;
  std::shared_ptr<WhereConstantCondConstantTensorYNode> Clone() const;
  const Tensor &GetCond() const noexcept;
  const Tensor &GetY() const noexcept;

private:
  const Tensor cond_;
  const Tensor y_;
};

} // namespace flow
} // namespace cpu_transformers

#endif
