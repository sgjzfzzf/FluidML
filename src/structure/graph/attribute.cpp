#include "structure/graph/attribute.h"
#include "exception/unreachable_exception.h"
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace graph {

Attribute::Attribute(cpu_transformers::Type data)
    : type_(Type::DataType), data_(data) {}

Attribute::Attribute(int64_t data) : type_(Type::Int64), data_(data) {}

Attribute::Attribute(float32_t data) : type_(Type::Float32), data_(data) {}

Attribute::Attribute(std::vector<int64_t> &&data)
    : type_(Type::Int64Array), data_(data) {}

Attribute::Attribute(Tensor &&data) : type_(Type::Tensor), data_(data) {}

Attribute::Type Attribute::GetType() const { return type_; }

cpu_transformers::Type Attribute::GetDataType() const {
#ifdef DEBUG
  assert(type_ == Type::DataType);
  assert(std::holds_alternative<cpu_transformers::Type>(data_));
#endif
  return std::get<cpu_transformers::Type>(data_);
}

int64_t Attribute::GetInt64() const {
#ifdef DEBUG
  assert(type_ == Type::Int64);
  assert(std::holds_alternative<int64_t>(data_));
#endif
  return std::get<int64_t>(data_);
}

float32_t Attribute::GetFloat32() const {
#ifdef DEBUG
  assert(type_ == Type::Float32);
  assert(std::holds_alternative<float32_t>(data_));
#endif
  return std::get<float32_t>(data_);
}

const std::vector<int64_t> &Attribute::GetInt64Array() const {
#ifdef DEBUG
  assert(type_ == Type::Int64Array);
  assert(std::holds_alternative<std::vector<int64_t>>(data_));
#endif
  return std::get<std::vector<int64_t>>(data_);
}

const Tensor &Attribute::GetTensor() const {
#ifdef DEBUG
  assert(type_ == Type::Tensor);
  assert(std::holds_alternative<Tensor>(data_));
#endif
  return std::get<Tensor>(data_);
}

const char *Attribute::TypeToString(Type type) {
  switch (type) {
  case Type::DataType:
    return "DataType";
  case Type::Int64:
    return "Int64";
  case Type::Float32:
    return "Float32";
  case Type::Int64Array:
    return "Int64Array";
  case Type::Tensor:
    return "Tensor";
  default:
#ifdef DEBUG
    throw UnreachableException();
#else
    return nullptr;
#endif
  }
}
} // namespace graph

} // namespace cpu_transformers
