#ifndef CPU_TRANSFORMERS_STRUCTURE_TENSOR_META_H_
#define CPU_TRANSFORMERS_STRUCTURE_TENSOR_META_H_

#include "utils/float.h"
#include "utils/type.h"
#include <cstddef>
#include <optional>
#include <vector>

namespace cpu_transformers {
class Meta {
public:
  Meta(Type type, std::vector<int64_t> &&shape);
  Meta(const Meta &meta) = default;
  Meta(Meta &&meta) = default;
  ~Meta() = default;
  Type GetType() const;
  const std::vector<int64_t> &GetShape() const;
  void AlignLeftTo(size_t size, float64_t value = 1.0);
  size_t GetElementsNum() const;
  size_t GetSize() const;
  friend bool operator==(const Meta &lhs, const Meta &rhs);
  friend bool operator!=(const Meta &lhs, const Meta &rhs);
  friend std::optional<Meta> BroadcastShape(Meta lhs, Meta rhs, Type type);
  friend std::optional<Meta> BroadcastMatMulShape(Meta lhs, Meta rhs,
                                                  Type type);
  friend std::optional<Meta> ReshapeShapeInference(Meta shape, size_t items);

private:
  const Type type_;
  std::vector<int64_t> shape_;
};
} // namespace cpu_transformers

#endif
