#ifndef CPU_TRANSFORMERS_STRUCTURE_GRAPH_NODE_H_
#define CPU_TRANSFORMERS_STRUCTURE_GRAPH_NODE_H_

#include "structure/graph/attribute.h"
#include <string>
#include <unordered_map>
namespace cpu_transformers {
namespace graph {
class Node {
public:
  // TODO: Cast, ConstantOfShape, Equal aren't in the current implementation
  // plan.
  enum class Op {
    Add,
    Cast,
    ConstantOfShape,
    Div,
    Equal,
    Erf,
    Gather,
    Gemm,
    LayerNormalization,
    MatMul,
    Mul,
    Pow,
    Reshape,
    Softmax,
    Split,
    Sub,
    Tanh,
    Transpose,
    Unsqueeze,
    Where,
  };
  Node(std::string &&name, Op op);
  Node(std::string &&name, Op op,
       std::unordered_map<std::string, Attribute> &&attributes);
  Node(const Node &node) = delete;
  Node(Node &&node) = default;
  const std::string &GetName() const;
  const Op &GetOp() const;
  bool HasAttribute(const std::string &name) const;
  const Attribute &GetAttribute(const std::string &name) const;
  const std::unordered_map<std::string, Attribute> &GetAttributes() const;

protected:
  const std::string name_;
  const Op op_;
  std::unordered_map<std::string, Attribute> attributes_;
};
} // namespace graph
} // namespace cpu_transformers

#endif
