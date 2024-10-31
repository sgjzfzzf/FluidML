#ifndef FLUIDML_WORKER_CONVERTER_H_
#define FLUIDML_WORKER_CONVERTER_H_

#include "structure/flow/flow.h"
#include "structure/graph/graph.h"
#include "worker/fwd.h"

namespace fluidml {
namespace worker {

class Converter {
public:
  virtual ~Converter() = default;
  virtual flow::Flow Run(const graph::Graph &graph) = 0;
  static std::unique_ptr<Converter> Make();

protected:
  Converter() = default;
  Converter(const Converter &converter) = delete;
  Converter(Converter &&converter) = default;
};

} // namespace worker
} // namespace fluidml

#endif
