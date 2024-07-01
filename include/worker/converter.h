#ifndef CPU_TRANSFORMERS_WORKER_CONVERTER_H_
#define CPU_TRANSFORMERS_WORKER_CONVERTER_H_

#include "structure/flow/flow.h"
#include "structure/graph/graph.h"

namespace cpu_transformers {
namespace worker {
class Converter {
public:
  Converter() = default;
  Converter(const Converter &converter) = delete;
  Converter(Converter &&converter) = delete;
  flow::Flow Run(const graph::Graph &graph);

private:
  void convertAddNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertAddDivErfAddMulMulNode(flow::Flow &flow,
                                     const graph::Graph &graph,
                                     const graph::Node &node);
  void convertDivNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertErfNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertGatherNode(flow::Flow &flow, const graph::Graph &graph,
                         const graph::Node &node);
  void convertGatherAddAddNode(flow::Flow &flow, const graph::Graph &graph,
                               const graph::Node &node);
  void convertGemmNode(flow::Flow &flow, const graph::Graph &graph,
                       const graph::Node &node);
  void convertLayerNormalizationNode(flow::Flow &flow,
                                     const graph::Graph &graph,
                                     const graph::Node &node);
  void convertMatMulNode(flow::Flow &flow, const graph::Graph &graph,
                         const graph::Node &node);
  void convertMulNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertPowNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertReshapeNode(flow::Flow &flow, const graph::Graph &graph,
                          const graph::Node &node);
  void convertSoftmaxNode(flow::Flow &flow, const graph::Graph &graph,
                          const graph::Node &node);
  void convertSplitNode(flow::Flow &flow, const graph::Graph &graph,
                        const graph::Node &node);
  void convertSubNode(flow::Flow &flow, const graph::Graph &graph,
                      const graph::Node &node);
  void convertTanhNode(flow::Flow &flow, const graph::Graph &graph,
                       const graph::Node &node);
  void convertTransposeNode(flow::Flow &flow, const graph::Graph &graph,
                            const graph::Node &node);
  void convertUnsqueezeNode(flow::Flow &flow, const graph::Graph &graph,
                            const graph::Node &node);
  void convertUnsqueezeSubMulNode(flow::Flow &flow, const graph::Graph &graph,
                                  const graph::Node &node);
  void convertWhereNode(flow::Flow &flow, const graph::Graph &graph,
                        const graph::Node &node);
};
} // namespace worker
} // namespace cpu_transformers

#endif
