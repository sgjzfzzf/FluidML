#include "structure/tensor/meta.h"
#include "utils/type.h"
#include <cstddef>
#include <numeric>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
Meta::Meta(Type type, std::vector<int64_t> &&shape)
    : type_(type), shape_(std::move(shape)) {}

Type Meta::GetType() const { return type_; }

const std::vector<int64_t> &Meta::GetShape() const { return shape_; }

void Meta::AlignLeftTo(size_t size, float64_t value) {
  size_t current_size = shape_.size();
  if (current_size >= size) {
    return;
  }
  shape_.insert(shape_.begin(), size - current_size, value);
}

size_t Meta::GetElementsNum() const {
#ifdef DEBUG
  for (int64_t dim : shape_) {
    assert(dim > 0);
  }
#endif
  return std::accumulate(shape_.begin(), shape_.end(), 1,
                         std::multiplies<int64_t>());
}

size_t Meta::GetSize() const {
  return GetElementsNum() * GetSizeFromType(type_);
}

bool operator==(const Meta &lhs, const Meta &rhs) {
  return lhs.type_ == rhs.type_ && lhs.shape_ == rhs.shape_;
}

bool operator!=(const Meta &lhs, const Meta &rhs) { return !(lhs == rhs); }

std::optional<Meta> BroadcastShape(Meta lhs, Meta rhs, Type type) {
  std::vector<int64_t> lhs_shape = lhs.GetShape();
  std::vector<int64_t> rhs_shape = rhs.GetShape();
  size_t lhs_size = lhs_shape.size();
  size_t rhs_size = rhs_shape.size();
  if (lhs_size > rhs_size) {
    rhs.AlignLeftTo(lhs_size);
  } else if (lhs_size < rhs_size) {
    lhs.AlignLeftTo(rhs_size);
  }
  lhs_shape = lhs.GetShape();
  rhs_shape = rhs.GetShape();
  lhs_size = lhs_shape.size();
  rhs_size = rhs_shape.size();
#ifdef DEBUG
  assert(lhs_shape.size() == rhs_shape.size());
#endif
  size_t aligned_size = lhs_shape.size();
  std::vector<int64_t> output_shape(aligned_size, 0);
  for (size_t i = 0; i < aligned_size; ++i) {
    if (lhs_shape[i] == rhs_shape[i]) {
      output_shape[i] = lhs_shape[i];
    } else if (lhs_shape[i] == 1) {
      output_shape[i] = rhs_shape[i];
    } else if (rhs_shape[i] == 1) {
      output_shape[i] = lhs_shape[i];
    } else {
      return std::nullopt;
    }
  }
  return Meta(type, std::move(output_shape));
}

std::optional<Meta> BroadcastMatMulShape(Meta lhs, Meta rhs, Type type) {
  std::vector<int64_t> lhs_shape = lhs.GetShape();
  std::vector<int64_t> rhs_shape = rhs.GetShape();
  size_t lhs_size = lhs_shape.size();
  size_t rhs_size = rhs_shape.size();
  if (lhs_size < 2 || rhs_size < 2) {
    return std::nullopt;
  }
  if (lhs_size < rhs_size) {
    lhs.AlignLeftTo(rhs_size);
  } else if (lhs_size > rhs_size) {
    rhs.AlignLeftTo(lhs_size);
  }
  lhs_shape = lhs.GetShape();
  rhs_shape = rhs.GetShape();
  lhs_size = lhs_shape.size();
  rhs_size = rhs_shape.size();
  size_t aligned_size = std::max(lhs_size, rhs_size);
#ifdef DEBUG
  assert(aligned_size == lhs_shape.size());
  assert(aligned_size == rhs_shape.size());
#endif
  size_t m = lhs_shape[aligned_size - 2];
  size_t k = lhs_shape[aligned_size - 1];
  size_t n = rhs_shape[aligned_size - 1];
  if (rhs_shape[aligned_size - 2] != k) {
    return std::nullopt;
  }
  std::vector<int64_t> output_shape(aligned_size, 0);
  for (size_t i = 0; i < aligned_size - 2; ++i) {
    if (lhs_shape[i] == rhs_shape[i]) {
      output_shape[i] = lhs_shape[i];
    } else if (lhs_shape[i] == 1) {
      output_shape[i] = rhs_shape[i];
    } else if (rhs_shape[i] == 1) {
      output_shape[i] = lhs_shape[i];
    } else {
      return std::nullopt;
    }
  }
  output_shape[aligned_size - 2] = m;
  output_shape[aligned_size - 1] = n;
  return Meta(type, std::move(output_shape));
}

std::optional<Meta> ReshapeShapeInference(Meta shape, size_t items) {
  int64_t neg_one = -1;
  std::vector<int64_t> shape_vec = shape.GetShape();
  size_t len = shape_vec.size();
  std::vector<int64_t> real_shape_vec(len, 0);
  size_t product = 1;
  for (size_t i = 0; i < len; ++i) {
    if (shape_vec[i] == -1) {
      if (neg_one != -1) {
        return std::nullopt;
      }
      neg_one = i;
    } else {
#ifdef DEBUG
      assert(shape_vec[i] > 0);
#endif
      real_shape_vec[i] = shape_vec[i];
      product *= shape_vec[i];
    }
  }
  if (neg_one != -1) {
    if (items % product != 0) {
      return std::nullopt;
    }
    real_shape_vec[neg_one] = items / product;
  }
  return Meta(shape.GetType(), std::move(real_shape_vec));
}
} // namespace cpu_transformers
