#include "optimization/graph/manager.h"
#include "worker/builder.h"
#include "worker/converter.h"
#include "worker/lower.h"
#include "worker/parser.h"
#include "worker/planner.h"
#include "worker/runner.h"
#include "gtest/gtest.h"

using namespace cpu_transformers;
using namespace cpu_transformers::context;
using namespace cpu_transformers::flow;
using namespace cpu_transformers::graph;
using namespace cpu_transformers::memory;
using namespace cpu_transformers::optimization;
using namespace cpu_transformers::worker;

TEST(ModelTest, BertTest) {
  Parser parser;
  GraphPassesManager pm;
  Converter converter;
  std::shared_ptr<Context> context = Context::Make();
  GeneralBuilder builder("bert", context);
  Lower lower(context);
  Runner runner(context);
  Graph graph = parser.Run(BERT_MODEL_PATH);
  pm.RegisterAllPasses();
  pm.Run(graph);
  Flow flow = converter.Run(graph);
  PlainLinearPlanner plain_linear_planner;
  PlainGreedyPlanner plain_greedy_planner;
  DPGreedyPlanner dp_greedy_planner;
  Sequence sequence = plain_greedy_planner.FlowToSequence(flow);
  // Sequence sequence = dp_greedy_planner.FlowToSequence(flow);
  Index plain_linear_index = plain_linear_planner.Run(sequence);
  Index greedy_index = plain_greedy_planner.Run(sequence);
  Index plain_dp_greedy_index = dp_greedy_planner.Run(sequence);
  ASSERT_LE(greedy_index.GetMaximum(), plain_linear_index.GetMaximum());
  builder.Run(sequence, greedy_index);
#ifdef DEBUG
  context->DumpModule("bert.mlir");
#endif
  lower.Run();
#ifdef DEBUG
  context->DumpModule("bert-llvm.mlir");
#endif
  std::vector<int64_t> input_ids(1 * 128, 0);
  std::vector<float32_t> attention_mask(1 * 128, 0), output0(1 * 128 * 768, 0),
      output1(1 * 768, 0);
  size_t time_cost = runner.Run(
      {
          {"input_ids", input_ids.data()},
          {"attention_mask", attention_mask.data()},
          {"onnx::Gather_1269", output0.data()},
          {"1272", output1.data()},
      },
      10);
  llvm::outs() << "Time cost: " << time_cost << "\n";
}
