#include "structure/memory/index.h"
#ifdef DEBUG
#include <cassert>
#endif

namespace fluidml {
namespace memory {
size_t Index::Get(const std::string &name) const {
  auto it = indices_.find(name);
#ifdef DEBUG
  assert(it != indices_.end());
#endif
  auto [index, size] = it->second;
  return index;
}

size_t Index::GetMaximum() const {
  size_t maximum = 0;
  for (const auto &[name, index_size] : indices_) {
    auto [index, size] = index_size;
    maximum = std::max(maximum, index + size);
  }
  return maximum;
}

void Index::Set(const std::string &name, size_t index, size_t size) {
  indices_.insert({name, std::make_tuple(index, size)});
}
} // namespace memory
} // namespace fluidml
