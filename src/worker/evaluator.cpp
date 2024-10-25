#include "worker/evaluator.h"
#include "evaluation/eval.h"
#include "nlohmann/json.hpp"
#include "structure/kernel/generator/generator.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <unordered_set>

namespace {

struct KeyKernelEvalHash {
  std::size_t operator()(
      const std::shared_ptr<cpu_transformers::evaluation::KernelEval> &eval)
      const {
    const cpu_transformers::kernel::KernelGenerator &generator =
        eval->GetKernelGenerator();
    return generator.GetShapeHashCode();
  }
};

struct KeyKernelEvalEqual {
  bool operator()(
      const std::shared_ptr<cpu_transformers::evaluation::KernelEval> &lhs,
      const std::shared_ptr<cpu_transformers::evaluation::KernelEval> &rhs)
      const {
    const cpu_transformers::kernel::KernelGenerator
        &lhs_generator = lhs->GetKernelGenerator(),
        &rhs_generator = rhs->GetKernelGenerator();
    return lhs_generator.ShapeEquals(rhs_generator);
  }
};

} // namespace

namespace ns {

void to_json(nlohmann::json &j,
             const cpu_transformers::worker::Evaluator &evaluator) {
  j = evaluator.ToJson();
}

} // namespace ns

namespace cpu_transformers {
namespace worker {

class EvaluatorImpl : public Evaluator {
public:
  EvaluatorImpl() = default;
  EvaluatorImpl(const EvaluatorImpl &evaluator) = delete;
  EvaluatorImpl(EvaluatorImpl &&evaluator_impl) = default;
  virtual ~EvaluatorImpl() = default;
  void RegisterEval(std::string &&name,
                    std::shared_ptr<evaluation::KernelEval> &&eval) override;
  evaluation::KernelEval &GetEval(const std::string &name) override;
  evaluation::SingleInputKernelEval &
  GetSingleInputEval(const std::string &name) override;
  evaluation::DoubleInputsKernelEval &
  GetDoubleInputsEval(const std::string &name) override;
  nlohmann::json ToJson() const override;

protected:
  std::unordered_set<std::shared_ptr<evaluation::KernelEval>, KeyKernelEvalHash,
                     KeyKernelEvalEqual>
      eval_set_;
  std::unordered_map<std::string, std::shared_ptr<evaluation::KernelEval>>
      eval_map_;
};

std::shared_ptr<Evaluator> Evaluator::Make() {
  return std::make_shared<EvaluatorImpl>();
}

std::ostream &operator<<(std::ostream &os, const Evaluator &evaluator) {
  os << evaluator.ToJson();
  return os;
}

void EvaluatorImpl::RegisterEval(
    std::string &&name, std::shared_ptr<evaluation::KernelEval> &&eval) {
  auto it = eval_set_.find(eval);
  if (it == eval_set_.end()) {
    eval_set_.insert(eval);
    eval_map_.insert_or_assign(std::move(name), eval);
  } else {
    eval_map_.insert_or_assign(std::move(name), *it);
  }
}

evaluation::KernelEval &EvaluatorImpl::GetEval(const std::string &name) {
  auto it = eval_map_.find(name);
#ifdef DEBUG
  assert(it != eval_map_.end());
#endif
  std::shared_ptr<evaluation::KernelEval> eval = it->second;
#ifdef DEBUG
  assert(eval != nullptr);
#endif
  return *eval;
}

evaluation::SingleInputKernelEval &
EvaluatorImpl::GetSingleInputEval(const std::string &name) {
  evaluation::KernelEval &eval = GetEval(name);
  evaluation::SingleInputKernelEval *single_input_eval =
      dynamic_cast<evaluation::SingleInputKernelEval *>(&eval);
#ifdef DEBUG
  assert(single_input_eval != nullptr);
#endif
  return *single_input_eval;
}

evaluation::DoubleInputsKernelEval &
EvaluatorImpl::GetDoubleInputsEval(const std::string &name) {
  evaluation::KernelEval &eval = GetEval(name);
  evaluation::DoubleInputsKernelEval *double_inputs_eval =
      dynamic_cast<evaluation::DoubleInputsKernelEval *>(&eval);
#ifdef DEBUG
  assert(double_inputs_eval != nullptr);
#endif
  return *double_inputs_eval;
}

nlohmann::json EvaluatorImpl::ToJson() const {
  nlohmann::json json;
  for (const std::shared_ptr<evaluation::KernelEval> &eval : eval_set_) {
    const kernel::KernelGenerator &generator = eval->GetKernelGenerator();
    const std::string kernel_name = generator.GetKernelName();
    nlohmann::json sub_json = eval->ToJson();
    if (json.contains(kernel_name)) {
      for (auto &elem : sub_json) {
        json[kernel_name].push_back(std::move(elem));
      }
    } else {
      json[kernel_name] = sub_json;
    }
  }
  return json;
}

} // namespace worker
} // namespace cpu_transformers
