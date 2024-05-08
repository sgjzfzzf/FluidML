#include "structure/memory/linear.h"

namespace cpu_transformers {
namespace memory {
Index LinearPlan::Run(Infos &infos) const {
  Index index;
  while (!infos.IsEmpty()) {
    Info info = infos.Pop();
    const std::string &name = info.GetName();
    const size_t maximum = index.GetMaximum();
    const size_t size = info.GetSize();
    index.Set(name, maximum, size);
  }
  return std::move(index);
}
} // namespace memory
} // namespace cpu_transformers