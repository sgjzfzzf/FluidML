#include "structure/tensor/tensor.h"
#include "structure/tensor/meta.h"
#include "utils/float.h"
#include "utils/hash.h"
#include <numeric>
#include <vector>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {

Tensor::Tensor(Meta &&meta) : meta_(meta) {
  data_.resize(std::accumulate(meta_.GetShape().begin(), meta_.GetShape().end(),
                               1, std::multiplies<int64_t>()));
}

Tensor::Tensor(Meta &&meta, std::vector<float64_t> &&data)
    : meta_(meta), data_(std::move(data)) {
#ifdef DEBUG
  assert(data_.size() == std::accumulate(meta_.GetShape().begin(),
                                         meta_.GetShape().end(), 1,
                                         std::multiplies<int64_t>()));
#endif
}

Tensor::Tensor(Type type, std::vector<int64_t> &&shape)
    : meta_(type, std::move(shape)) {}

Tensor::Tensor(Type type, std::vector<int64_t> &&shape,
               std::vector<float64_t> &&data)
    : Tensor(Meta(type, std::move(shape)), std::move(data)) {}

const std::vector<float64_t> &Tensor::Get() const { return data_; }

float64_t &Tensor::Get(const std::vector<size_t> &indices) {
  return const_cast<float64_t &>(getImpl(indices));
}

const float64_t &Tensor::Get(const std::vector<size_t> &indices) const {
  return getImpl(indices);
}

const Meta &Tensor::GetMeta() const { return meta_; }

Type Tensor::GetType() const { return meta_.GetType(); }

const std::vector<int64_t> &Tensor::GetShape() const {
  return meta_.GetShape();
}

size_t Tensor::GetHashCode() const {
  size_t hash = meta_.GetHashCode();
  std::hash<float64_t> f64_hash;
  for (float64_t value : data_) {
    hash ^= f64_hash(value) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  return hash;
}

bool operator==(const Tensor &lhs, const Tensor &rhs) {
  return lhs.meta_ == rhs.meta_ && lhs.data_ == rhs.data_;
}

bool operator!=(const Tensor &lhs, const Tensor &rhs) { return !(lhs == rhs); }

const float64_t &Tensor::getImpl(const std::vector<size_t> &indices) const {
  int64_t index = 0;
  const std::vector<int64_t> &shape = meta_.GetShape();
  for (int64_t i = 0; i < indices.size(); ++i) {
#ifdef DEBUG
    assert(indices[i] < shape[i]);
#endif
    index = index * shape[i] + indices[i];
  }
  return data_[index];
}

} // namespace cpu_transformers
