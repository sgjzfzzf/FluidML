#ifndef FLUIDML_TENSOR_TENSOR_H_
#define FLUIDML_TENSOR_TENSOR_H_

#include "structure/tensor/meta.h"
#include "utils/float.h"
#include "utils/type.h"
#include <vector>

namespace fluidml {
class Tensor {

public:
  Tensor(Meta &&meta);
  Tensor(Meta &&meta, std::vector<float64_t> &&data);
  Tensor(Type type, std::vector<int64_t> &&shape);
  Tensor(Type type, std::vector<int64_t> &&shape,
         std::vector<float64_t> &&data);
  Tensor(const Tensor &) = default;
  Tensor(Tensor &&) = default;
  ~Tensor() = default;
  Tensor &operator=(const Tensor &) = default;
  Tensor &operator=(Tensor &&) = default;
  const std::vector<float64_t> &Get() const;
  float64_t &Get(const std::vector<size_t> &indices);
  const float64_t &Get(const std::vector<size_t> &indices) const;
  const Meta &GetMeta() const;
  Type GetType() const;
  const std::vector<int64_t> &GetShape() const;
  size_t GetHashCode() const;
  friend bool operator==(const Tensor &lhs, const Tensor &rhs);
  friend bool operator!=(const Tensor &lhs, const Tensor &rhs);

private:
  const float64_t &getImpl(const std::vector<size_t> &indices) const;

  Meta meta_;
  std::vector<float64_t> data_;
};

} // namespace fluidml

#endif
