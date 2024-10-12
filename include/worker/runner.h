#ifndef CPU_TRANSFORMERS_WORKER_RUNNER_H_
#define CPU_TRANSFORMERS_WORKER_RUNNER_H_

#ifdef BUILD_PYTHON
#include "pybind11/numpy.h"
#endif
#include "structure/context/context.h"
#include "worker/fwd.h"

namespace cpu_transformers {
namespace worker {

class Runner {
public:
  Runner(std::shared_ptr<context::Context> context = nullptr);
  Runner(const Runner &runner) = delete;
  Runner(Runner &&runner) = default;
  virtual ~Runner() = default;
  size_t Run(const std::unordered_map<std::string, void *> &args,
             size_t epoch = 1);
#ifdef BUILD_PYTHON
  size_t Run(const std::unordered_map<std::string, pybind11::array> &args);
#endif

private:
  std::shared_ptr<context::Context> context_;
};

} // namespace worker
} // namespace cpu_transformers

#endif
