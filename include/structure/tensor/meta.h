#ifndef FLUIDML_STRUCTURE_TENSOR_META_H_
#define FLUIDML_STRUCTURE_TENSOR_META_H_

#include "utils/float.h"
#include "utils/type.h"
#include <cstddef>
#include <optional>
#include <vector>

namespace fluidml {
class Meta {
public:
  Meta(Type type, std::vector<int64_t> &&shape);
  Meta(const Meta &meta) = default;
  Meta(Meta &&meta) = default;
  Meta &operator=(const Meta &meta) = default;
  Meta &operator=(Meta &&meta) = default;
  ~Meta() = default;
  Type GetType() const;
  const std::vector<int64_t> &GetShape() const;
  Meta AlignLeftTo(size_t size, float64_t value = 1.0) const;
  size_t GetElementsNum() const;
  size_t GetSize() const;
  size_t GetHashCode() const;
  friend bool operator==(const Meta &lhs, const Meta &rhs);
  friend bool operator!=(const Meta &lhs, const Meta &rhs);
  friend std::optional<Meta> BroadcastShape(const Meta &lhs, const Meta &rhs,
                                            Type type);
  friend std::optional<Meta> BroadcastMatMulShape(const Meta &lhs,
                                                  const Meta &rhs, Type type);
  friend std::optional<Meta> ReshapeShapeInference(const Meta &shape,
                                                   size_t items);

private:
  Type type_;
  std::vector<int64_t> shape_;
};
} // namespace fluidml

#endif
