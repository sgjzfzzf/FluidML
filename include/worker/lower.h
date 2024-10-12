#ifndef CPU_TRANSFORMERS_WORKER_LOWER_H_
#define CPU_TRANSFORMERS_WORKER_LOWER_H_

#include "mlir/Pass/PassManager.h"
#include "structure/context/context.h"
#include "worker/fwd.h"

namespace cpu_transformers {
namespace worker {
class Lower {
public:
  Lower(std::shared_ptr<context::Context> context = nullptr);
  Lower(const Lower &lower) = delete;
  Lower(Lower &&lower) = delete;
  virtual ~Lower() = default;
  void Run();

private:
  std::shared_ptr<context::Context> context_;
  mlir::PassManager pm_;
};
} // namespace worker
} // namespace cpu_transformers

#endif
