#ifndef FLUIDML_STRUCTURE_GRAPH_ATTRIBUTE_H_
#define FLUIDML_STRUCTURE_GRAPH_ATTRIBUTE_H_

#include "structure/graph/fwd.h"
#include "structure/tensor/tensor.h"
#include "utils/float.h"
#include "utils/type.h"
#include <cstdint>
#include <variant>
#include <vector>

namespace fluidml {
namespace graph {
class Attribute {
public:
  enum class Type {
    DataType,
    Int64,
    Float32,
    Int64Array,
    Tensor,
  };

  Attribute(fluidml::Type data);
  Attribute(int64_t data);
  Attribute(float32_t data);
  Attribute(std::vector<int64_t> &&data);
  Attribute(Tensor &&data);
  Attribute(const Attribute &attribute) = default;
  Attribute(Attribute &&attribute) = default;
  Type GetType() const;
  fluidml::Type GetDataType() const;
  int64_t GetInt64() const;
  float32_t GetFloat32() const;
  const std::vector<int64_t> &GetInt64Array() const;
  const Tensor &GetTensor() const;
  static const char *TypeToString(Type type);

private:
  const Type type_;
  const std::variant<fluidml::Type, int64_t, float32_t, std::vector<int64_t>,
                     Tensor>
      data_;
};
} // namespace graph
} // namespace fluidml

#endif
