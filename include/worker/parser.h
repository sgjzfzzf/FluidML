#ifndef FLUIDML_WORKER_PARSER_H_
#define FLUIDML_WORKER_PARSER_H_

#include "onnx/onnx_pb.h"
#include "structure/graph/graph.h"
#include "worker/fwd.h"

namespace fluidml {
namespace worker {

class Parser {
public:
  virtual ~Parser() = default;
  graph::Graph Run(std::string_view input);
  graph::Graph Run(std::istream &input);
  virtual graph::Graph Run(onnx::ModelProto &model_proto) = 0;
  static std::unique_ptr<Parser> Make();

protected:
  Parser() = default;
  Parser(const Parser &parser) = delete;
  Parser(Parser &&parser) = default;
};

} // namespace worker
} // namespace fluidml

#endif
