#include "benchmark/benchmark.h"
#include "fmt/core.h"
#include "utils/float.h"
#include "worker/executor.h"
#ifdef USE_LOGS
#define GLOG_USE_GLOG_EXPORT
#include "glog/flags.h"
#include "glog/logging.h"
#endif

static void BM_RunBertModel(benchmark::State &state) {
  using namespace fluidml;
  for (auto _ : state) {
    std::string name = "convbert";
    std::string input = CONVBERT_MODEL_PATH,
                mlir = fmt::format("{}.mlir", name),
                llvm = fmt::format("{}-llvm.mlir", name),
                json = fmt::format("{}.json", name);
    std::unique_ptr<worker::Executor> executor =
        worker::Executor::MakeDPGreedy(std::move(name));
    executor->Compile(input, mlir, llvm, json);
    std::vector<int64_t> input_ids(1 * 128, 0);
    std::vector<float32_t> attention_mask(1 * 128, 0), output(1 * 128 * 768, 0);
    const size_t time_cost = executor->Invoke({
        {"input_ids", input_ids.data()},
        {"attention_mask", attention_mask.data()},
        {"2625", output.data()},
    });
#ifdef USE_LOGS
    LOG(INFO) << "Time cost: " << time_cost << " ns\n";
#endif
  }
}

BENCHMARK(BM_RunBertModel);

int main(int argc, char **argv) {
#ifdef USE_LOGS
  google::InitGoogleLogging(argv[0]);
  google::SetLogDestination(google::GLOG_INFO, "benchmark_log_");
#endif
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
#ifdef USE_LOGS
  google::ShutdownGoogleLogging();
#endif
  return 0;
}
