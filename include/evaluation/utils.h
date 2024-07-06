#ifndef CPU_TRANSFORMERS_EVALUATION_UTILS_H_
#define CPU_TRANSFORMERS_EVALUATION_UTILS_H_

#include "structure/tensor/meta.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cpu_transformers {
namespace evaluation {

std::vector<uint8_t> FillBuffer(const Meta &meta);
std::vector<uint8_t> RandomFillBuffer(const Meta &meta);
std::vector<std::vector<size_t>> GenAllOrders(size_t len);
std::vector<int64_t> GenPhysicalShape(const std::vector<int64_t> &shape,
                                      const std::vector<size_t> &layout);
std::vector<int64_t> GenStrides(const std::vector<int64_t> &shape);
std::vector<int64_t> GenStrides(const std::vector<int64_t> &shape,
                                const std::vector<size_t> &layout);

} // namespace evaluation
} // namespace cpu_transformers

#endif
