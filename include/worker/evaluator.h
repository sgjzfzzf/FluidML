#ifndef FLUIDML_WORKER_EVALUATOR_H_
#define FLUIDML_WORKER_EVALUATOR_H_

#include "evaluation/eval.h"
#include "nlohmann/json_fwd.hpp"
#include "worker/fwd.h"
#include <memory>

namespace ns {
void to_json(nlohmann::json &json, const fluidml::worker::Evaluator &evaluator);
}

namespace fluidml {
namespace worker {

class Evaluator {
public:
  virtual ~Evaluator() = default;
  virtual void RegisterEval(std::string &&name,
                            std::shared_ptr<evaluation::KernelEval> &&eval) = 0;
  virtual evaluation::KernelEval &GetEval(const std::string &name) = 0;
  virtual evaluation::SingleInputKernelEval &
  GetSingleInputEval(const std::string &name) = 0;
  virtual evaluation::DoubleInputsKernelEval &
  GetDoubleInputsEval(const std::string &name) = 0;
  virtual nlohmann::json ToJson() const = 0;
  static std::shared_ptr<Evaluator> Make();
  friend std::ostream &operator<<(std::ostream &os, const Evaluator &evaluator);

protected:
  Evaluator() = default;
  Evaluator(const Evaluator &evaluator) = delete;
  Evaluator(Evaluator &&evaluator) = default;
};

} // namespace worker
} // namespace fluidml

#endif
