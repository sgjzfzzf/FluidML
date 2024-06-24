#ifndef CPU_TRANSFORMERS_OPTIMIZATION_GRAPH_MANAGER_H_
#define CPU_TRANSFORMERS_OPTIMIZATION_GRAPH_MANAGER_H_

#include "optimization/graph/pass.h"
#include "optimization/manager.h"
#include "structure/graph/graph.h"
#include <initializer_list>
#include <list>

namespace cpu_transformers {
namespace optimization {

class GraphPassesManager : public PassesManager {
public:
  GraphPassesManager() = default;
  GraphPassesManager(std::initializer_list<std::shared_ptr<GraphPass>> passes);
  GraphPassesManager(const GraphPassesManager &manager) = default;
  GraphPassesManager(GraphPassesManager &&manager) = default;
  void RegisterAllPasses();
  void Run(graph::Graph &graph) const;

private:
  std::list<std::shared_ptr<GraphPass>> passes_;
};

} // namespace optimization
} // namespace cpu_transformers

#endif
