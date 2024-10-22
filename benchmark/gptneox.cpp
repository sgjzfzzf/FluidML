#include "benchmark/benchmark.h"
#include "fmt/core.h"
#include "utils/float.h"
#include "worker/executor.h"
#ifdef USE_LOGS
#define GLOG_USE_GLOG_EXPORT
#include "glog/flags.h"
#include "glog/logging.h"
#endif

static void BM_RunGptNeoxModel(benchmark::State &state) {
  using namespace cpu_transformers;
  for (auto _ : state) {
    std::string name = "gptneox";
    std::string input = GPTNEOX_MODEL_PATH, mlir = fmt::format("{}.mlir", name),
                llvm = fmt::format("{}-llvm.mlir", name);
    std::unique_ptr<worker::Executor> executor =
        worker::Executor::MakeDPGreedy(std::move(name));
    executor->Compile(input, mlir, llvm);
    std::vector<int64_t> input_ids(1 * 128, 0);
    std::vector<float32_t> attention_mask(1 * 128, 0), output0(1 * 128 * 32, 0),
        output1(1 * 4 * 128 * 8, 0), output2(1 * 4 * 128 * 8, 0),
        output3(1 * 4 * 128 * 8, 0), output4(1 * 4 * 128 * 8, 0),
        output5(1 * 4 * 128 * 8, 0), output6(1 * 4 * 128 * 8, 0),
        output7(1 * 4 * 128 * 8, 0), output8(1 * 4 * 128 * 8, 0),
        output9(1 * 4 * 128 * 8, 0), output10(1 * 4 * 128 * 8, 0);
    const size_t time_cost = executor->Invoke({
        {"input_ids", input_ids.data()},
        {"attention_mask", attention_mask.data()},
        {"1182", output0.data()},
        {"key", output1.data()},
        {"value", output2.data()},
        {"key.3", output3.data()},
        {"value.3", output4.data()},
        {"key.7", output5.data()},
        {"value.7", output6.data()},
        {"key.11", output7.data()},
        {"value.11", output8.data()},
        {"key.15", output9.data()},
        {"value.15", output10.data()},
    });
#ifdef USE_LOGS
    LOG(INFO) << "Time cost: " << time_cost << " ns\n";
#endif
  }
}

BENCHMARK(BM_RunGptNeoxModel);

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
