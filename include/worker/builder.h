#ifndef CPU_TRANSFORMERS_WORKER_BUILDERS_H_
#define CPU_TRANSFORMERS_WORKER_BUILDERS_H_

#include "structure/context/context.h"
#include "structure/memory/index.h"
#include "worker/scheduler.h"

namespace cpu_transformers {
namespace worker {
class Builder {
public:
  Builder(std::string &&function_name = "main",
          std::shared_ptr<context::Context> context = nullptr);
  Builder(const Builder &builder) = delete;
  Builder(Builder &&builder) = delete;
  virtual ~Builder() = default;
  void Run(const flow::Sequence &sequence, const memory::Index &index);

private:
  virtual Scheduler &getScheduler() = 0;
  const std::string function_name_;
  std::shared_ptr<context::Context> context_;
};

class NaiveBuilder : public Builder {
public:
  NaiveBuilder(std::string &&function_name = "main",
               std::shared_ptr<context::Context> context = nullptr);
  NaiveBuilder(const NaiveBuilder &naiveBuilder) = delete;
  NaiveBuilder(NaiveBuilder &&naiveBuilder) = delete;
  ~NaiveBuilder() = default;

private:
  Scheduler &getScheduler() override;
  NaiveScheduler scheduler_;
};
} // namespace worker
} // namespace cpu_transformers

#endif
