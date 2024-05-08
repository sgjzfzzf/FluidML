#ifndef CPU_TRANSFORMERS_STRUCTURE_FLOW_NODE_H_
#define CPU_TRANSFORMERS_STRUCTURE_FLOW_NODE_H_

#include "structure/flow/edge.h"
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
  virtual const std::string &GetName() const noexcept;
  virtual size_t GetBufferSize() const noexcept;

protected:
  const std::string name_;
};

class AddNode : public Node {
public:
  AddNode(std::string &&name);
  AddNode(const AddNode &node) = delete;
  AddNode(AddNode &&node) = default;
  virtual ~AddNode() = default;
};

class AddConstantNode : public AddNode {
public:
  AddConstantNode(std::string &&name, std::shared_ptr<Edge> &&input,
                  std::shared_ptr<Edge> &&output);
  AddConstantNode(const AddConstantNode &node) = delete;
  AddConstantNode(AddConstantNode &&node) = default;
  virtual ~AddConstantNode() = default;
  virtual Type GetType() const noexcept = 0;
  std::shared_ptr<Edge> GetInput() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

protected:
  std::shared_ptr<Edge> input_;
  std::shared_ptr<Edge> output_;
};

class AddConstantScalarNode : public AddConstantNode {
public:
  AddConstantScalarNode(std::string &&name, Type type, float64_t value,
                        std::shared_ptr<Edge> &&input,
                        std::shared_ptr<Edge> &&output);
  AddConstantScalarNode(const AddConstantScalarNode &node) = delete;
  AddConstantScalarNode(AddConstantScalarNode &&node) = default;
  virtual ~AddConstantScalarNode() = default;
  Type GetType() const noexcept override;
  float64_t GetValue() const noexcept;

private:
  const Type type_;
  const float64_t value_;
};

class AddConstantTensorNode : public AddConstantNode {
public:
  AddConstantTensorNode(std::string &&name, Tensor &&tensor,
                        std::shared_ptr<Edge> &&input,
                        std::shared_ptr<Edge> &&output);
  AddConstantTensorNode(const AddConstantTensorNode &node) = delete;
  AddConstantTensorNode(AddConstantTensorNode &&node) = default;
  virtual ~AddConstantTensorNode() = default;
  Type GetType() const noexcept override;
  const Tensor &GetTensor() const noexcept;

private:
  const Tensor tensor_;
};

class AddCommonNode : public AddNode {
public:
  AddCommonNode(std::string &&name, std::shared_ptr<Edge> &&lhs,
                std::shared_ptr<Edge> &&rhs, std::shared_ptr<Edge> &&output);
  AddCommonNode(const AddCommonNode &node) = delete;
  AddCommonNode(AddCommonNode &&node) = default;
  virtual ~AddCommonNode() = default;
  std::shared_ptr<Edge> GetLhs() const noexcept;
  std::shared_ptr<Edge> GetRhs() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  std::shared_ptr<Edge> lhs_;
  std::shared_ptr<Edge> rhs_;
  std::shared_ptr<Edge> output_;
};

class DivNode : public Node {
public:
  DivNode(std::string &&name);
  DivNode(const DivNode &node) = delete;
  DivNode(DivNode &&node) = default;
  virtual ~DivNode() = default;
};

class DivConstantScalarNode : public DivNode {
public:
  DivConstantScalarNode(std::string &&name, Type type, float64_t value,
                        std::shared_ptr<Edge> &&input,
                        std::shared_ptr<Edge> &&output);
  DivConstantScalarNode(const DivConstantScalarNode &node) = delete;
  DivConstantScalarNode(DivConstantScalarNode &&node) = default;
  virtual ~DivConstantScalarNode() = default;
  Type GetType() const noexcept;
  float64_t GetValue() const noexcept;
  std::shared_ptr<Edge> GetInput() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  const Type type_;
  const float64_t value_;
  std::shared_ptr<Edge> input_;
  std::shared_ptr<Edge> output_;
};

class ErfNode : public Node {
public:
  ErfNode(std::string &&name, std::shared_ptr<Edge> &&input,
          std::shared_ptr<Edge> &&output);
  ErfNode(const ErfNode &node) = delete;
  ErfNode(ErfNode &&node) = default;
  virtual ~ErfNode() = default;
  std::shared_ptr<Edge> GetInput() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  std::shared_ptr<Edge> input_;
  std::shared_ptr<Edge> output_;
};

class GatherNode : public Node {
public:
  constexpr static int64_t kAxis = 0;
  constexpr static const char *kAxisAttrName = "axis";
  GatherNode(std::string &&name, std::shared_ptr<Edge> &&output,
             int64_t axis = kAxis);
  GatherNode(const GatherNode &node) = delete;
  GatherNode(GatherNode &&node) = default;
  virtual ~GatherNode() = default;
  std::shared_ptr<Edge> GetOutput() const noexcept;
  int64_t GetAxis() const noexcept;

protected:
  std::shared_ptr<Edge> output_;
  const int64_t axis_;
};

class GatherConstantIndexScalarNode : public GatherNode {
public:
  GatherConstantIndexScalarNode(std::string &&name, std::shared_ptr<Edge> &&lhs,
                                int64_t rhs, std::shared_ptr<Edge> &&output,
                                int64_t axis = kAxis);
  GatherConstantIndexScalarNode(const GatherConstantIndexScalarNode &node) =
      delete;
  GatherConstantIndexScalarNode(GatherConstantIndexScalarNode &&node) = default;
  virtual ~GatherConstantIndexScalarNode() = default;
  std::shared_ptr<Edge> GetLhs() const noexcept;
  int64_t GetRhs() const noexcept;

private:
  std::shared_ptr<Edge> lhs_;
  const int64_t rhs_;
};

class GatherConstantDataTensorNode : public GatherNode {
public:
  GatherConstantDataTensorNode(std::string &&name, Tensor &&lhs,
                               std::shared_ptr<Edge> &&rhs,
                               std::shared_ptr<Edge> &&output,
                               int64_t axis = kAxis);
  GatherConstantDataTensorNode(const GatherConstantDataTensorNode &node) =
      delete;
  GatherConstantDataTensorNode(GatherConstantDataTensorNode &&node) = default;
  virtual ~GatherConstantDataTensorNode() = default;
  const Tensor &GetLhs() const noexcept;
  std::shared_ptr<Edge> GetRhs() const noexcept;

private:
  const Tensor lhs_;
  std::shared_ptr<Edge> rhs_;
};

class GemmNode : public Node {
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

class GemmConstantWeightsBiasNode : public GemmNode {
public:
  GemmConstantWeightsBiasNode(std::string &&name, std::shared_ptr<Edge> &&input,
                              Tensor &&weights, Tensor &&bias,
                              std::shared_ptr<Edge> &&output, float64_t alpha,
                              float64_t beta, bool transA, bool transB);
  GemmConstantWeightsBiasNode(const GemmConstantWeightsBiasNode &node) = delete;
  GemmConstantWeightsBiasNode(GemmConstantWeightsBiasNode &&node) = default;
  virtual ~GemmConstantWeightsBiasNode() = default;
  std::shared_ptr<Edge> GetInput() const noexcept;
  const Tensor &GetWeights() const noexcept;
  const Tensor &GetBias() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  std::shared_ptr<Edge> input_;
  const Tensor weights_;
  const Tensor bias_;
  std::shared_ptr<Edge> output_;
};

// TODO: Implement the kernrl for the LayerNormalization and its derivatives.
class LayerNormalizationNode : public Node {
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

class LayerNormalizationConstantScaleBiasNode : public LayerNormalizationNode {
public:
  LayerNormalizationConstantScaleBiasNode(std::string &&name,
                                          std::shared_ptr<Edge> &&input,
                                          Tensor &&scale, Tensor &&bias,
                                          std::shared_ptr<Edge> &&output,
                                          int64_t axis, float64_t epsilon);
  LayerNormalizationConstantScaleBiasNode(
      const LayerNormalizationConstantScaleBiasNode &node) = delete;
  LayerNormalizationConstantScaleBiasNode(
      LayerNormalizationConstantScaleBiasNode &&node) = default;
  virtual ~LayerNormalizationConstantScaleBiasNode() = default;
  size_t GetBufferSize() const noexcept override;
  const Meta &GetMeta() const noexcept override;
  std::shared_ptr<Edge> GetInput() const noexcept;
  const Tensor &GetScale() const noexcept;
  const Tensor &GetBias() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  std::shared_ptr<Edge> input_;
  const Tensor scale_;
  const Tensor bias_;
  std::shared_ptr<Edge> output_;
};

class MatMulNode : public Node {
public:
  MatMulNode(std::string &&name);
  MatMulNode(const MatMulNode &node) = delete;
  MatMulNode(MatMulNode &&node) = default;
  virtual ~MatMulNode() = default;
};

class MatMulConstantLhsNode : public MatMulNode {
public:
  MatMulConstantLhsNode(std::string &&name, Tensor &&lhs,
                        std::shared_ptr<Edge> &&rhs,
                        std::shared_ptr<Edge> &&output);
  MatMulConstantLhsNode(const MatMulConstantLhsNode &node) = delete;
  MatMulConstantLhsNode(MatMulConstantLhsNode &&node) = default;
  virtual ~MatMulConstantLhsNode() = default;
  const Tensor &GetLhs() const noexcept;
  std::shared_ptr<Edge> GetRhs() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  const Tensor lhs_;
  std::shared_ptr<Edge> rhs_;
  std::shared_ptr<Edge> output_;
};

class MatMulConstantRhsNode : public MatMulNode {
public:
  MatMulConstantRhsNode(std::string &&name, std::shared_ptr<Edge> &&lhs,
                        Tensor &&rhs, std::shared_ptr<Edge> &&output);
  MatMulConstantRhsNode(const MatMulConstantRhsNode &node) = delete;
  MatMulConstantRhsNode(MatMulConstantRhsNode &&node) = default;
  virtual ~MatMulConstantRhsNode() = default;
  std::shared_ptr<Edge> GetLhs() const noexcept;
  const Tensor &GetRhs() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  std::shared_ptr<Edge> lhs_;
  const Tensor rhs_;
  std::shared_ptr<Edge> output_;
};

class MatMulCommonNode : public MatMulNode {
public:
  MatMulCommonNode(std::string &&name, std::shared_ptr<Edge> &&lhs,
                   std::shared_ptr<Edge> &&rhs, std::shared_ptr<Edge> &&output);
  MatMulCommonNode(const MatMulCommonNode &node) = delete;
  MatMulCommonNode(MatMulCommonNode &&node) = default;
  virtual ~MatMulCommonNode() = default;
  std::shared_ptr<Edge> GetLhs() const noexcept;
  std::shared_ptr<Edge> GetRhs() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  std::shared_ptr<Edge> lhs_;
  std::shared_ptr<Edge> rhs_;
  std::shared_ptr<Edge> output_;
};

class MulNode : public Node {
public:
  MulNode(std::string &&name);
  MulNode(const MulNode &node) = delete;
  MulNode(MulNode &&node) = default;
  virtual ~MulNode() = default;
};

class MulConstantNode : public MulNode {
public:
  MulConstantNode(std::string &&name, std::shared_ptr<Edge> &&input,
                  std::shared_ptr<Edge> &&output);
  MulConstantNode(const MulConstantNode &node) = delete;
  MulConstantNode(MulConstantNode &&node) = default;
  virtual ~MulConstantNode() = default;
  std::shared_ptr<Edge> GetInput() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

protected:
  std::shared_ptr<Edge> input_;
  std::shared_ptr<Edge> output_;
};

class MulConstantScalarNode : public MulConstantNode {
public:
  MulConstantScalarNode(std::string &&name, std::shared_ptr<Edge> &&input,
                        Type type, float64_t value,
                        std::shared_ptr<Edge> &&output);
  MulConstantScalarNode(const MulConstantScalarNode &node) = delete;
  MulConstantScalarNode(MulConstantScalarNode &&node) = default;
  virtual ~MulConstantScalarNode() = default;
  Type GetType() const noexcept;
  float64_t GetValue() const noexcept;

private:
  const Type type_;
  const float64_t value_;
};

class MulConstantTensorNode : public MulConstantNode {
public:
  MulConstantTensorNode(std::string &&name, std::shared_ptr<Edge> &&input,
                        Tensor &&tensor, std::shared_ptr<Edge> &&output);
  MulConstantTensorNode(const MulConstantTensorNode &node) = delete;
  MulConstantTensorNode(MulConstantTensorNode &&node) = default;
  virtual ~MulConstantTensorNode() = default;
  const Tensor &GetTensor() const noexcept;

private:
  const Tensor tensor_;
};

class MulCommonNode : public MulNode {
public:
  MulCommonNode(std::string &&name, std::shared_ptr<Edge> &&lhs,
                std::shared_ptr<Edge> &&rhs, std::shared_ptr<Edge> &&output);
  MulCommonNode(const MulCommonNode &node) = delete;
  MulCommonNode(MulCommonNode &&node) = default;
  virtual ~MulCommonNode() = default;
  std::shared_ptr<Edge> GetLhs() const noexcept;
  std::shared_ptr<Edge> GetRhs() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  std::shared_ptr<Edge> lhs_;
  std::shared_ptr<Edge> rhs_;
  std::shared_ptr<Edge> output_;
};

class PowNode : public Node {
public:
  PowNode(std::string &&name, std::shared_ptr<Edge> &&input, Type type,
          float64_t exp, std::shared_ptr<Edge> &&output);
  PowNode(const PowNode &node) = delete;
  PowNode(PowNode &&node) = default;
  virtual ~PowNode() = default;
  std::shared_ptr<Edge> GetInput() const noexcept;
  Type GetType() const noexcept;
  float64_t GetExp() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  std::shared_ptr<Edge> input_;
  const Type type_;
  const float64_t exp_;
  std::shared_ptr<Edge> output_;
};

class ReshapeNode : public Node {
public:
  ReshapeNode(std::string &&name, std::shared_ptr<Edge> &&input,
              std::shared_ptr<Edge> &&output);
  ReshapeNode(const ReshapeNode &node) = delete;
  ReshapeNode(ReshapeNode &&node) = default;
  virtual ~ReshapeNode() = default;
  std::shared_ptr<Edge> GetInput() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  std::shared_ptr<Edge> input_;
  std::shared_ptr<Edge> output_;
};

class SoftmaxNode : public Node {
public:
  constexpr static int64_t kAxis = -1;
  constexpr static const char *kAxisAttrName = "axis";
  SoftmaxNode(std::string &&name, std::shared_ptr<Edge> &&input,
              std::shared_ptr<Edge> &&output, int64_t axis = kAxis);
  SoftmaxNode(const SoftmaxNode &node) = delete;
  SoftmaxNode(SoftmaxNode &&node) = default;
  virtual ~SoftmaxNode() = default;
  size_t GetBufferSize() const noexcept override;
  std::shared_ptr<Edge> GetInput() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;
  int64_t GetAxis() const noexcept;
  const Meta &GetMeta() const noexcept;

private:
  std::shared_ptr<Edge> input_;
  std::shared_ptr<Edge> output_;
  const int64_t axis_;
};

class SplitNode : public Node {
public:
  constexpr static int64_t kAxis = 0;
  constexpr static const char *kAxisAttrName = "axis";
  SplitNode(std::string &&name, std::shared_ptr<Edge> &&input,
            std::vector<std::shared_ptr<Edge>> &&outputs, int64_t axis);
  SplitNode(const SplitNode &node) = delete;
  SplitNode(SplitNode &&node) = default;
  virtual ~SplitNode() = default;
  std::shared_ptr<Edge> GetInput() const noexcept;
  const std::vector<std::shared_ptr<Edge>> &GetOutputs() const noexcept;
  int64_t GetAxis() const noexcept;
  const Meta &GetMeta() const noexcept;

private:
  std::shared_ptr<Edge> input_;
  std::vector<std::shared_ptr<Edge>> outputs_;
  const int64_t axis_;
};

class SubNode : public Node {
public:
  SubNode(std::string &&name);
  SubNode(const SubNode &node) = delete;
  SubNode(SubNode &&node) = default;
  virtual ~SubNode() = default;
};

class SubConstantScalarLhsNode : public SubNode {
public:
  SubConstantScalarLhsNode(std::string &&name, std::shared_ptr<Edge> &&input,
                           Type type, float64_t value,
                           std::shared_ptr<Edge> &&output);
  SubConstantScalarLhsNode(const SubConstantScalarLhsNode &node) = delete;
  SubConstantScalarLhsNode(SubConstantScalarLhsNode &&node) = default;
  virtual ~SubConstantScalarLhsNode() = default;
  std::shared_ptr<Edge> GetInput() const noexcept;
  Type GetType() const noexcept;
  float64_t GetValue() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  std::shared_ptr<Edge> input_;
  const Type type_;
  const float64_t value_;
  std::shared_ptr<Edge> output_;
};

class TanhNode : public Node {
public:
  TanhNode(std::string &&name, std::shared_ptr<Edge> &&input,
           std::shared_ptr<Edge> &&output);
  TanhNode(const TanhNode &node) = delete;
  TanhNode(TanhNode &&node) = default;
  virtual ~TanhNode() = default;
  std::shared_ptr<Edge> GetInput() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  std::shared_ptr<Edge> input_;
  std::shared_ptr<Edge> output_;
};

class TransposeNode : public Node {
public:
  constexpr static const char *kPermAttrName = "perm";
  TransposeNode(std::string &&name, std::shared_ptr<Edge> &&input,
                std::shared_ptr<Edge> &&output, std::vector<int64_t> &&perm);
  TransposeNode(const TransposeNode &node) = delete;
  TransposeNode(TransposeNode &&node) = default;
  virtual ~TransposeNode() = default;
  std::shared_ptr<Edge> GetInput() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;
  const std::vector<int64_t> &GetPerm() const noexcept;

private:
  std::shared_ptr<Edge> input_;
  std::shared_ptr<Edge> output_;
  const std::vector<int64_t> perm_;
};

class UnsqueezeNode : public Node {
public:
  UnsqueezeNode(std::string &&name, std::shared_ptr<Edge> &&input,
                std::shared_ptr<Edge> &&output, std::vector<int64_t> &&axes);
  UnsqueezeNode(const UnsqueezeNode &node) = delete;
  UnsqueezeNode(UnsqueezeNode &&node) = default;
  virtual ~UnsqueezeNode() = default;
  std::shared_ptr<Edge> GetInput() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;
  const std::vector<int64_t> &GetAxes() const noexcept;

private:
  std::shared_ptr<Edge> input_;
  std::shared_ptr<Edge> output_;
  const std::vector<int64_t> axes_;
};

class WhereNode : public Node {
public:
  WhereNode(std::string &&name);
  WhereNode(const WhereNode &node) = delete;
  WhereNode(WhereNode &&node) = default;
  virtual ~WhereNode() = default;
};

class WhereConstantCondConstantScalarYNode : public WhereNode {
public:
  WhereConstantCondConstantScalarYNode(std::string &&name, Tensor &&cond,
                                       std::shared_ptr<Edge> &&x, Type type,
                                       float64_t y,
                                       std::shared_ptr<Edge> &&output);
  WhereConstantCondConstantScalarYNode(
      const WhereConstantCondConstantScalarYNode &node) = delete;
  WhereConstantCondConstantScalarYNode(
      WhereConstantCondConstantScalarYNode &&node) = default;
  virtual ~WhereConstantCondConstantScalarYNode() = default;
  const Tensor &GetCond() const noexcept;
  std::shared_ptr<Edge> GetX() const noexcept;
  Type GetType() const noexcept;
  float64_t GetY() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  const Tensor cond_;
  std::shared_ptr<Edge> x_;
  const Type type_;
  const float64_t y_;
  std::shared_ptr<Edge> output_;
};

class WhereConstantCondConstantTensorYNode : public WhereNode {
public:
  WhereConstantCondConstantTensorYNode(std::string &&name, Tensor &&cond,
                                       std::shared_ptr<Edge> &&x, Tensor &&y,
                                       std::shared_ptr<Edge> &&output);
  WhereConstantCondConstantTensorYNode(
      const WhereConstantCondConstantTensorYNode &node) = delete;
  WhereConstantCondConstantTensorYNode(
      WhereConstantCondConstantTensorYNode &&node) = default;
  virtual ~WhereConstantCondConstantTensorYNode() = default;
  const Tensor &GetCond() const noexcept;
  std::shared_ptr<Edge> GetX() const noexcept;
  const Tensor &GetY() const noexcept;
  std::shared_ptr<Edge> GetOutput() const noexcept;

private:
  const Tensor cond_;
  std::shared_ptr<Edge> x_;
  const Tensor y_;
  std::shared_ptr<Edge> output_;
};

} // namespace flow
} // namespace cpu_transformers

#endif
