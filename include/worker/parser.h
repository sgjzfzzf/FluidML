#ifndef CPU_TRANSFORMERS_WORKER_PARSER_H_
#define CPU_TRANSFORMERS_WORKER_PARSER_H_

#include "structure/graph/graph.h"
#include "worker/fwd.h"

namespace cpu_transformers {
namespace worker {
class Parser {
public:
  Parser() = default;
  Parser(const Parser &parser) = delete;
  Parser(Parser &&parser) = delete;
  graph::Graph Run(const std::string &file);
};
} // namespace worker
} // namespace cpu_transformers

#endif
