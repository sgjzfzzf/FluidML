#ifndef CPU_TRANSFORMERS_WORKER_SCHEDULER_H_
#define CPU_TRANSFORMERS_WORKER_SCHEDULER_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "structure/flow/sequence.h"
#include "worker/worker.h"
#include <unordered_map>

namespace cpu_transformers {
namespace worker {

class Scheduler {
public:
  Scheduler() = default;
  Scheduler(const Scheduler &scheduler) = delete;
  Scheduler(Scheduler &&scheduler) = default;
  virtual ~Scheduler() = default;
  virtual void Run(mlir::OpBuilder &builder, const flow::Sequence &sequence,
                   std::unordered_map<std::string, mlir::Value> &symbol_table);
};

} // namespace worker
} // namespace cpu_transformers

#endif
