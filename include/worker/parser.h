#ifndef CPU_TRANSFORMERS_WORKER_PARSER_H_
#define CPU_TRANSFORMERS_WORKER_PARSER_H_

#include "structure/graph/graph.h"
#include "worker/fwd.h"

namespace cpu_transformers {
namespace worker {

class Parser {
public:
  virtual ~Parser() = default;
  virtual graph::Graph Run(const std::string &file) = 0;
  static std::unique_ptr<Parser> Make();

protected:
  Parser() = default;
  Parser(const Parser &parser) = delete;
  Parser(Parser &&parser) = delete;
};

} // namespace worker
} // namespace cpu_transformers

#endif
