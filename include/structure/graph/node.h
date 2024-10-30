#ifndef CPU_TRANSFORMERS_STRUCTURE_GRAPH_NODE_H_
#define CPU_TRANSFORMERS_STRUCTURE_GRAPH_NODE_H_

#include "structure/graph/attribute.h"
#include "structure/graph/fwd.h"
#include <string>
#include <unordered_map>

namespace cpu_transformers {
namespace graph {

class Node {
public:
  // TODO: ConstantOfShape aren't in the current implementation plan.
  enum class Op {
    Add,
    AddDivErfAddMulMul,
    Cast,
    Concat,
    ConstantOfShape,
    Conv,
    CumSum,
    Div,
    Equal,
    Erf,
    Gather,
    GatherAddAdd,
    Gemm,
    LayerNormalization,
    MatMul,
    Mul,
    Neg,
    Not,
    Pad,
    Pow,
    ReduceMean,
    Reshape,
    Slice,
    Softmax,
    Sqrt,
    Sub,
    Tanh,
    Transpose,
    Unsqueeze,
    UnsqueezeSubMul,
    Where,
  };
  Node(std::string &&name, Op op,
       std::unordered_map<std::string, Attribute> &&attributes = {},
       Graph *graph = nullptr);
  Node(const Node &node) = delete;
  Node(Node &&node) = default;
  const std::string &GetName() const;
  const Op &GetOp() const;
  bool HasAttribute(const std::string &name) const;
  const Attribute &GetAttribute(const std::string &name) const;
  const std::unordered_map<std::string, Attribute> &GetAttributes() const;
  Graph *GetGraph() const;
  std::vector<std::shared_ptr<Edge>> GetInputEdges() const;
  std::vector<std::shared_ptr<Edge>> GetOutputEdges() const;
  std::vector<std::shared_ptr<Node>> GetInputNodes() const;
  std::vector<std::shared_ptr<Node>> GetOutputNodes() const;
  void Delete();
  void ClearInput(const Edge &edge);
  void ClearInput(const std::string &name);
  void ClearOutput(const Edge &edge);
  void ClearOutput(const std::string &name);
  void ClearInputs();
  void ClearOutputs();
  void PutInput(Edge &edge);
  void PutOutput(Edge &edge);
  friend class Graph;

protected:
  const std::string name_;
  const Op op_;
  std::unordered_map<std::string, Attribute> attributes_;
  Graph *graph_;
};

} // namespace graph
} // namespace cpu_transformers

#endif
