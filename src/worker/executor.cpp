#include "worker/executor.h"
#include "structure/context/context.h"
#include "worker/builder.h"
#include "worker/converter.h"
#include "worker/lower.h"
#include "worker/parser.h"
#include "worker/planner.h"
#include "worker/runner.h"
#include <memory>

namespace fluidml {
namespace worker {

class ExecutorImpl : public Executor {
public:
  ExecutorImpl(std::string &&name, size_t epoch = 1);
  ExecutorImpl(const ExecutorImpl &) = delete;
  ExecutorImpl(ExecutorImpl &&) = default;
  virtual ~ExecutorImpl() = default;
  void Compile(std::istream &input, std::ofstream *mlir, std::ofstream *llvm,
               std::ofstream *json) override;
  void Compile(std::string_view inpupt,
               std::optional<std::string_view> mlir = std::nullopt,
               std::optional<std::string_view> llvm = std::nullopt,
               std::optional<std::string_view> json = std::nullopt) override;
  size_t Invoke(const std::unordered_map<std::string, void *> &args) override;
#ifdef BUILD_PYTHON
  size_t
  Invoke(const std::unordered_map<std::string, pybind11::array> &args) override;
#endif

protected:
  virtual std::unique_ptr<worker::GeneralBuilder> makeBuilder() = 0;
  virtual std::unique_ptr<worker::Planner> makePlanner() = 0;
  context::Context context_;
  const std::string name_;
  const size_t epoch_;
};

class PlainLinearExecutor : public ExecutorImpl {
public:
  using ExecutorImpl::ExecutorImpl;
  PlainLinearExecutor(const PlainLinearExecutor &) = delete;
  PlainLinearExecutor(PlainLinearExecutor &&) = default;
  virtual ~PlainLinearExecutor() = default;

private:
  std::unique_ptr<worker::GeneralBuilder> makeBuilder() override;
  std::unique_ptr<worker::Planner> makePlanner() override;
};

class PlainGreedyExecutor : public ExecutorImpl {
public:
  using ExecutorImpl::ExecutorImpl;
  PlainGreedyExecutor(const PlainGreedyExecutor &) = delete;
  PlainGreedyExecutor(PlainGreedyExecutor &&) = default;
  virtual ~PlainGreedyExecutor() = default;

private:
  std::unique_ptr<worker::GeneralBuilder> makeBuilder() override;
  std::unique_ptr<worker::Planner> makePlanner() override;
};

class DPGreedyExecutor : public ExecutorImpl {
public:
  using ExecutorImpl::ExecutorImpl;
  DPGreedyExecutor(const DPGreedyExecutor &) = delete;
  DPGreedyExecutor(DPGreedyExecutor &&) = default;
  virtual ~DPGreedyExecutor() = default;

private:
  std::unique_ptr<worker::GeneralBuilder> makeBuilder() override;
  std::unique_ptr<worker::Planner> makePlanner() override;
};

std::unique_ptr<Executor> Executor::MakePlainLinear(std::string &&name,
                                                    size_t epoch) {
  return std::make_unique<PlainLinearExecutor>(std::move(name), epoch);
}

std::unique_ptr<Executor> Executor::MakePlainGreedy(std::string &&name,
                                                    size_t epoch) {
  return std::make_unique<PlainGreedyExecutor>(std::move(name), epoch);
}

std::unique_ptr<Executor> Executor::MakeDPGreedy(std::string &&name,
                                                 size_t epoch) {
  return std::make_unique<DPGreedyExecutor>(std::move(name), epoch);
}

void ExecutorImpl::Compile(std::istream &input, std::ofstream *mlir,
                           std::ofstream *llvm, std::ofstream *json) {
  {
    std::unique_ptr<worker::Parser> parser = worker::Parser::Make();
    std::unique_ptr<worker::Converter> converter = worker::Converter::Make();
    std::unique_ptr<worker::GeneralBuilder> builder = makeBuilder();
    std::unique_ptr<worker::Lower> lower = context_.MakeLower();
    graph::Graph graph = parser->Run(input);
    flow::Flow flow = converter->Run(graph);
    std::unique_ptr<worker::Planner> planner = makePlanner();
    auto [sequence, index, dp_json] = planner->Run(flow);
    if (json) {
      *json << dp_json;
    }
    builder->Run(sequence, index);
    if (mlir) {
      *mlir << context_;
    }
    lower->Run();
    if (llvm) {
      *llvm << context_;
    }
  }
}

void ExecutorImpl::Compile(std::string_view input,
                           std::optional<std::string_view> mlir,
                           std::optional<std::string_view> llvm,
                           std::optional<std::string_view> json) {
  std::ifstream ifs(input.data());
  std::ofstream mlir_ofs, llvm_ofs, json_ofs;
  if (mlir) {
    mlir_ofs.open(mlir->data());
  }
  if (llvm) {
    llvm_ofs.open(llvm->data());
  }
  if (json) {
    json_ofs.open(json->data());
  }
  Compile(ifs, mlir ? &mlir_ofs : nullptr, llvm ? &llvm_ofs : nullptr,
          json ? &json_ofs : nullptr);
}

size_t
ExecutorImpl::Invoke(const std::unordered_map<std::string, void *> &args) {
  std::unique_ptr<worker::Runner> runner = context_.MakeRunner();
  return runner->Run(args, epoch_);
}

#ifdef BUILD_PYTHON
size_t ExecutorImpl::Invoke(
    const std::unordered_map<std::string, pybind11::array> &args) {
  std::unique_ptr<worker::Runner> runner = context_.MakeRunner();
  return runner->Run(args, epoch_);
}
#endif

ExecutorImpl::ExecutorImpl(std::string &&name, size_t epoch)
    : name_(std::move(name)), epoch_(epoch) {}

std::unique_ptr<worker::GeneralBuilder> PlainLinearExecutor::makeBuilder() {
  std::string name = name_;
  return context_.MakePlainGeneralBuilder(std::move(name));
}

std::unique_ptr<worker::Planner> PlainLinearExecutor::makePlanner() {
  return context_.MakePlainLinearPlanner();
}

std::unique_ptr<worker::GeneralBuilder> PlainGreedyExecutor::makeBuilder() {
  std::string name = name_;
  return context_.MakePlainGeneralBuilder(std::move(name));
}

std::unique_ptr<worker::Planner> PlainGreedyExecutor::makePlanner() {
  return context_.MakePlainGreedyPlanner();
}

std::unique_ptr<worker::GeneralBuilder> DPGreedyExecutor::makeBuilder() {
  std::string name = name_;
  return context_.MakeDPGeneralBuilder(std::move(name));
}

std::unique_ptr<worker::Planner> DPGreedyExecutor::makePlanner() {
  return context_.MakeDPGreedyPlanner();
}

} // namespace worker
} // namespace fluidml
