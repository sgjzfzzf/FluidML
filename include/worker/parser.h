#ifndef CPU_TRANSFORMERS_WORKER_PARSER_H_
#define CPU_TRANSFORMERS_WORKER_PARSER_H_
#include "onnx/onnx_pb.h"
#include "structure/graph/graph.h"

namespace cpu_transformers {
namespace worker {
class Parser {
public:
  Parser() = default;
  Parser(const Parser &parser) = delete;
  Parser(Parser &&parser) = delete;
  graph::Graph Run(const std::string &file);
  graph::Graph Run(onnx::ModelProto &model_proto);
};
} // namespace worker
} // namespace cpu_transformers

#endif
