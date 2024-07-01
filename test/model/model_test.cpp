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
  NaiveBuilder builder("bert", context);
  Lower lower(context);
  Runner runner(context);
  Graph graph = parser.Run(BERT_MODEL_PATH);
  pm.RegisterAllPasses();
  pm.Run(graph);
  Flow flow = converter.Run(graph);
  LinearPlanner linear_planner;
  GreedyPlanner greedy_planner;
  Sequence sequence = greedy_planner.FlowToSequence(flow);
  Index linear_index = linear_planner.Run(sequence);
  Index greedy_index = greedy_planner.Run(sequence);
  ASSERT_LE(greedy_index.GetMaximum(), linear_index.GetMaximum());
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
  runner.Run({
      {"input_ids", input_ids.data()},
      {"attention_mask", attention_mask.data()},
      {"onnx::Gather_1269", output0.data()},
      {"1272", output1.data()},
  });
}

TEST(ModelTest, GPT2Test) {
  Parser parser;
  Converter converter;
  std::shared_ptr<Context> context = Context::Make();
  NaiveBuilder builder("gpt2", context);
  Lower lower(context);
  Graph graph = parser.Run(GPT2_MODEL_PATH);
  Flow flow = converter.Run(graph);
  LinearPlanner planner;
  Sequence sequence = planner.FlowToSequence(flow);
  Index index = planner.Run(sequence);
  builder.Run(sequence, index);
#ifdef DEBUG
  context->DumpModule("gpt2.mlir");
#endif
  lower.Run();
#ifdef DEBUG
  context->DumpModule("gpt2-llvm.mlir");
#endif
}
