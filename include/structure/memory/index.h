#ifndef CPU_TRANSFORMERS_STRUCTURE_MEMORY_INDEX_H_
#define CPU_TRANSFORMERS_STRUCTURE_MEMORY_INDEX_H_

#include <cstddef>
#include <string>
#include <unordered_map>

namespace cpu_transformers {
namespace memory {

class Index {
public:
  Index() = default;
  Index(const Index &) = delete;
  Index(Index &&) = default;
  ~Index() = default;
  size_t Get(const std::string &name) const;
  size_t GetMaximum() const;
  void Set(const std::string &name, size_t index, size_t size);

private:
  std::unordered_map<std::string, std::tuple<size_t, size_t>> indices_;
};

} // namespace memory
} // namespace cpu_transformers

#endif