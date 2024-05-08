#ifndef CPU_TRANSFORMERS_STRUCTURE_GRAPH_EDGE_H_
#define CPU_TRANSFORMERS_STRUCTURE_GRAPH_EDGE_H_

#include "structure/tensor/meta.h"
#include "structure/tensor/tensor.h"
#include "utils/type.h"
#include <string>
#include <vector>

namespace cpu_transformers {
namespace graph {
class Edge {

public:
  Edge(std::string &&name);
  Edge(const Edge &edge) = delete;
  Edge(Edge &&edge) = default;
  virtual ~Edge() = default;
  const std::string &GetName() const;
  virtual Type GetType() const = 0;
  virtual const std::vector<int64_t> &GetShape() const = 0;

protected:
  std::string name_;
};

class ConstantEdge : public Edge {
public:
  ConstantEdge(std::string &&name);
  ConstantEdge(const ConstantEdge &edge) = delete;
  ConstantEdge(ConstantEdge &&edge) = default;
};

class ConstantScalarEdge : public ConstantEdge {
public:
  ConstantScalarEdge(std::string &&name, Type type, float64_t scalar);
  ConstantScalarEdge(const ConstantScalarEdge &edge) = delete;
  ConstantScalarEdge(ConstantScalarEdge &&edge) = default;
  Type GetType() const override;
  const std::vector<int64_t> &GetShape() const override;
  float64_t GetValue() const;

private:
  Type type_;
  float64_t scalar_;
};

class ConstantTensorEdge : public ConstantEdge {
public:
  ConstantTensorEdge(std::string &&name, Type type, Tensor &&tensor);
  ConstantTensorEdge(const ConstantTensorEdge &edge) = delete;
  ConstantTensorEdge(ConstantTensorEdge &&edge) = default;
  Type GetType() const override;
  const std::vector<int64_t> &GetShape() const override;
  const Tensor &GetValue() const;

private:
  Tensor tensor_;
};

class NonConstantEdge : public Edge {
public:
  NonConstantEdge(std::string &&name, Type type, std::vector<int64_t> &&shape);
  NonConstantEdge(const NonConstantEdge &edge) = delete;
  NonConstantEdge(NonConstantEdge &&edge) = default;
  const Meta &GetMeta() const;
  Type GetType() const override;
  const std::vector<int64_t> &GetShape() const override;

private:
  Meta meta_;
};

class PureEdge : public NonConstantEdge {
public:
  PureEdge(std::string &&name, Type type, std::vector<int64_t> &&shape);
  PureEdge(const PureEdge &edge) = delete;
  PureEdge(PureEdge &&edge) = default;
};

class InputEdge : public NonConstantEdge {
public:
  InputEdge(std::string &&name, Type type, std::vector<int64_t> &&shape);
  InputEdge(const InputEdge &edge) = delete;
  InputEdge(InputEdge &&edge) = default;
};

class OutputEdge : public NonConstantEdge {
public:
  OutputEdge(std::string &&name, Type type, std::vector<int64_t> &&shape);
  OutputEdge(const OutputEdge &edge) = delete;
  OutputEdge(OutputEdge &&edge) = default;
};
} // namespace graph

} // namespace cpu_transformers

#endif
