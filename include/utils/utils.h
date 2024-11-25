#ifndef FLUIDML_UTILS_H_
#define FLUIDML_UTILS_H_

#include "structure/tensor/meta.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fluidml {
namespace utils {

std::vector<uint8_t> FillBuffer(const Meta &meta);
std::vector<uint8_t> RandomFillBuffer(const Meta &meta);
std::vector<std::vector<size_t>> GenAllOrders(size_t len);
std::vector<std::vector<size_t>>
GenAllIndicesInOrder(const std::vector<int64_t> &shape);
std::vector<int64_t> GenPhysicalShape(const std::vector<int64_t> &shape,
                                      const std::vector<size_t> &layout);
std::vector<int64_t> GenStrides(const std::vector<int64_t> &shape);
std::vector<int64_t> GenStrides(const std::vector<int64_t> &shape,
                                const std::vector<size_t> &layout);
std::vector<size_t> GenDefaultLayout(size_t shape_len);
std::vector<size_t> GenDefaultLayout(const std::vector<int64_t> &shape);
size_t GenIndex(const std::vector<size_t> &indices,
                const std::vector<int64_t> &strides);

} // namespace utils
} // namespace fluidml

#endif
