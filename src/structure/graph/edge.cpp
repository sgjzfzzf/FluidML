#include "structure/graph/edge.h"
#include "structure/tensor/meta.h"
#include "utils/type.h"

namespace cpu_transformers {
namespace graph {
Edge::Edge(std::string &&name) : name_(name) {}

const std::string &Edge::GetName() const { return name_; }

ConstantEdge::ConstantEdge(std::string &&name) : Edge(std::move(name)) {}

ConstantScalarEdge::ConstantScalarEdge(std::string &&name, Type type,
                                       float64_t scalar)
    : ConstantEdge(std::move(name)), type_(type), scalar_(scalar) {}

Type ConstantScalarEdge::GetType() const { return type_; }

const std::vector<int64_t> &ConstantScalarEdge::GetShape() const {
  static const std::vector<int64_t> shape = {};
  return shape;
}

float64_t ConstantScalarEdge::GetValue() const { return scalar_; }

ConstantTensorEdge::ConstantTensorEdge(std::string &&name, Type type,
                                       Tensor &&tensor)
    : ConstantEdge(std::move(name)), tensor_(std::move(tensor)) {}

Type ConstantTensorEdge::GetType() const { return tensor_.GetType(); }

const std::vector<int64_t> &ConstantTensorEdge::GetShape() const {
  return tensor_.GetShape();
}

const Tensor &ConstantTensorEdge::GetValue() const { return tensor_; }

NonConstantEdge::NonConstantEdge(std::string &&name, Type type,
                                 std::vector<int64_t> &&shape)
    : Edge(std::move(name)), meta_(type, std::move(shape)) {}

const Meta &NonConstantEdge::GetMeta() const { return meta_; }

Type NonConstantEdge::GetType() const { return GetMeta().GetType(); }

const std::vector<int64_t> &NonConstantEdge::GetShape() const {
  return GetMeta().GetShape();
}

PureEdge::PureEdge(std::string &&name, Type type, std::vector<int64_t> &&shape)
    : NonConstantEdge(std::move(name), type, std::move(shape)) {}

InputEdge::InputEdge(std::string &&name, Type type,
                     std::vector<int64_t> &&shape)
    : NonConstantEdge(std::move(name), type, std::move(shape)) {}

OutputEdge::OutputEdge(std::string &&name, Type type,
                       std::vector<int64_t> &&shape)
    : NonConstantEdge(std::move(name), type, std::move(shape)) {}
} // namespace graph
} // namespace cpu_transformers
