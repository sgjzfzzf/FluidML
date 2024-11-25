#include "utils/utils.h"
#include <cstddef>
#include <random>
#ifdef DEBUG
#include <cassert>
#endif

namespace fluidml {
namespace utils {

std::vector<uint8_t> FillBuffer(const Meta &meta) {
  return std::vector<uint8_t>(meta.GetSize(), 0);
}

std::vector<uint8_t> RandomFillBuffer(const Meta &meta) {
  std::vector<uint8_t> buffer = FillBuffer(meta);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dis(0, 255);
  for (size_t i = 0; i < meta.GetSize(); ++i) {
    buffer[i] = dis(gen);
  }
  return buffer;
}

std::vector<std::vector<size_t>> GenAllOrders(size_t len) {
  std::vector<std::vector<size_t>> orders = {};
  if (len == 0) {
    orders.push_back({});
  } else {
    std::vector<std::vector<size_t>> prev_orders = GenAllOrders(len - 1);
    for (const std::vector<size_t> &prev_order : prev_orders) {
      for (size_t i = 0; i < len; ++i) {
        std::vector<size_t> order = prev_order;
        order.insert(order.begin() + i, len - 1);
        orders.push_back(std::move(order));
      }
    }
  }
  return orders;
}

std::vector<std::vector<size_t>>
GenAllIndicesInOrder(const std::vector<int64_t> &shape) {
  std::vector<std::vector<size_t>> results = {};
  if (shape.size() == 0) {
    results = {{}};
  } else {
    const size_t last_size = shape.back();
    std::vector<int64_t> shape_slice(shape.begin(), shape.end() - 1);
    std::vector<std::vector<size_t>> prev_results =
        GenAllIndicesInOrder(shape_slice);
    for (std::vector<size_t> prev_result : prev_results) {
      for (size_t i = 0; i < last_size; ++i) {
        std::vector<size_t> result = prev_result;
        result.push_back(i);
        results.push_back(std::move(result));
      }
    }
  }
  return results;
}

std::vector<int64_t> GenPhysicalShape(const std::vector<int64_t> &shape,
                                      const std::vector<size_t> &layout) {
  const size_t shape_len = shape.size();
  std::vector<int64_t> physical_shape(shape_len);
  for (size_t i = 0; i < shape_len; ++i) {
    const size_t j = layout[i];
#ifdef DEBUG
    assert(j < shape_len);
#endif
    physical_shape[j] = shape[i];
  }
  return physical_shape;
}

std::vector<int64_t> GenStrides(const std::vector<int64_t> &shape) {
  const int64_t shape_len = shape.size();
  std::vector<int64_t> strides(shape_len);
  strides[shape_len - 1] = 1;
  for (int64_t i = shape_len - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

std::vector<int64_t> GenStrides(const std::vector<int64_t> &shape,
                                const std::vector<size_t> &layout) {
  const size_t shape_len = shape.size();
  std::vector<int64_t> physical_shape = GenPhysicalShape(shape, layout),
                       physical_strides = GenStrides(physical_shape),
                       strides(shape_len);
  for (size_t i = 0; i < shape_len; ++i) {
    const size_t j = layout[i];
#ifdef DEBUG
    assert(j < shape_len);
#endif
    strides[i] = physical_strides[j];
  }
  return strides;
}

std::vector<size_t> GenDefaultLayout(size_t shape_len) {
  std::vector<size_t> layout(shape_len);
  for (size_t i = 0; i < shape_len; ++i) {
    layout[i] = i;
  }
  return layout;
}

std::vector<size_t> GenDefaultLayout(const std::vector<int64_t> &shape) {
  const size_t shape_len = shape.size();
  return GenDefaultLayout(shape_len);
}

size_t GenIndex(const std::vector<size_t> &indices,
                const std::vector<int64_t> &strides) {
  size_t index = 0;
  const size_t indices_len = indices.size(), strides_len = strides.size();
  for (size_t i = 0; i < indices_len; ++i) {
    index += indices[i] * strides[i];
  }
  return index;
}

} // namespace utils
} // namespace fluidml
