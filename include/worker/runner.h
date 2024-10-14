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
  virtual ~Runner() = default;
  virtual size_t Run(const std::unordered_map<std::string, void *> &args,
                     size_t epoch = 1) = 0;
#ifdef BUILD_PYTHON
  virtual size_t
  Run(const std::unordered_map<std::string, pybind11::array> &args,
      size_t epoch) = 0;
#endif
  static std::unique_ptr<Runner> Make(context::Context &&context);

protected:
  Runner() = default;
  Runner(const Runner &runner) = delete;
  Runner(Runner &&runner) = default;
  context::Context context_;
};

} // namespace worker
} // namespace cpu_transformers

#endif
