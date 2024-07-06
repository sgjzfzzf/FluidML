#ifndef CPU_TRANSFORMERS_WORKER_EVALUATOR_H_
#define CPU_TRANSFORMERS_WORKER_EVALUATOR_H_

#include "evaluation/eval.h"
#include <memory>

namespace cpu_transformers {
namespace worker {

class Evaluator {
public:
  virtual void RegisterEval(std::string &&name,
                            std::shared_ptr<evaluation::KernelEval> &&eval) = 0;
  virtual evaluation::KernelEval &GetEval(const std::string &name) = 0;
  virtual evaluation::SingleInputKernelEval &
  GetSingleInputEval(const std::string &name) = 0;
  virtual evaluation::DoubleInputsKernelEval &
  GetDoubleInputsEval(const std::string &name) = 0;
  static std::shared_ptr<Evaluator> Make();

protected:
  Evaluator() = default;
  Evaluator(const Evaluator &evaluator) = delete;
  Evaluator(Evaluator &&evaluator) = default;
};

} // namespace worker
} // namespace cpu_transformers

#endif
