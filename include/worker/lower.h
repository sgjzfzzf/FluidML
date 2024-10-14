#ifndef CPU_TRANSFORMERS_WORKER_LOWER_H_
#define CPU_TRANSFORMERS_WORKER_LOWER_H_

#include "structure/context/context.h"
#include "worker/fwd.h"
#include <memory>

namespace cpu_transformers {
namespace worker {

class Lower {
public:
  virtual ~Lower() = default;
  virtual void Run() = 0;
  static std::unique_ptr<Lower> Make(context::Context &&context);

protected:
  Lower() = default;
  Lower(const Lower &lower) = delete;
  Lower(Lower &&lower) = delete;
};

} // namespace worker
} // namespace cpu_transformers

#endif
