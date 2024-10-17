#include "benchmark/benchmark.h"
#include "optimization/graph/manager.h"
#include "worker/builder.h"
#include "worker/converter.h"
#include "worker/lower.h"
#include "worker/parser.h"
#include "worker/planner.h"
#include "worker/runner.h"
#ifdef DEBUG
#include "fmt/core.h"
#include <fstream>
#endif
#ifdef USE_LOGS
#define GLOG_USE_GLOG_EXPORT
#include "glog/flags.h"
#include "glog/logging.h"
#endif

const std::tuple<std::string, std::string> kModelInfo[] = {
    {"bert", BERT_MODEL_PATH},
};

static void BM_RunBertModel(benchmark::State &state) {
  const int index = state.range(0);
  auto [name, path] = kModelInfo[index];
  for (auto _ : state) {
    std::unique_ptr<cpu_transformers::worker::Parser> parser =
        cpu_transformers::worker::Parser::Make();
    cpu_transformers::optimization::GraphPassesManager pm;
    std::unique_ptr<cpu_transformers::worker::Converter> converter =
        cpu_transformers::worker::Converter::Make();
    cpu_transformers::context::Context context;
    std::unique_ptr<cpu_transformers::worker::GeneralBuilder> builder =
        context.MakeGeneralBuilder(name.c_str());
    std::unique_ptr<cpu_transformers::worker::Lower> lower =
        context.MakeLower();
    std::unique_ptr<cpu_transformers::worker::Runner> runner =
        context.MakeRunner();
    cpu_transformers::graph::Graph graph = parser->Run(path);
    pm.RegisterAllPasses();
    pm.Run(graph);
    cpu_transformers::flow::Flow flow = converter->Run(graph);
    std::unique_ptr<cpu_transformers::worker::PlainGreedyPlanner>
        plain_greedy_planner = context.MakePlainGreedyPlanner();
    std::unique_ptr<cpu_transformers::worker::DPGreedyPlanner>
        dp_greedy_planner = context.MakeDPGreedyPlanner();
    // cpu_transformers::flow::Sequence sequence =
    //     plain_greedy_planner->FlowToSequence(flow);
    cpu_transformers::flow::Sequence sequence =
        dp_greedy_planner->FlowToSequence(flow);
    cpu_transformers::memory::Index greedy_index =
        plain_greedy_planner->Run(sequence);
    builder->Run(sequence, greedy_index);
#ifdef DEBUG
    std::ofstream ofs(fmt::format("{}.mlir", name));
    ofs << context;
    ofs.close();
#endif
    lower->Run();
#ifdef DEBUG
    ofs.open(fmt::format("{}-llvm.mlir", name));
    ofs << context;
    ofs.close();
#endif
    std::vector<int64_t> input_ids(1 * 128, 0);
    std::vector<cpu_transformers::float32_t> attention_mask(1 * 128, 0),
        output0(1 * 128 * 768, 0), output1(1 * 768, 0);
    size_t time_cost = runner->Run(
        {
            {"input_ids", input_ids.data()},
            {"attention_mask", attention_mask.data()},
            {"onnx::Gather_1269", output0.data()},
            {"1272", output1.data()},
        },
        1);
#ifdef USE_LOGS
    LOG(INFO) << "Time cost: " << time_cost << " ns\n";
#endif
  }
}

BENCHMARK(BM_RunBertModel)->Arg(0);

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
