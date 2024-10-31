#ifndef FLUIDML_STRUCTURE_CONTEXT_FACTORY_H_
#define FLUIDML_STRUCTURE_CONTEXT_FACTORY_H_

#include "structure/flow/node.h"
#include "structure/kernel/generator/generator.h"
#include <memory>

namespace fluidml {
namespace context {

class Factory {
public:
  virtual ~Factory() = default;
  virtual std::shared_ptr<kernel::KernelGenerator>
  MakeKernelGenerator(const flow::Node &node) = 0;
  static std::unique_ptr<Factory> Make();

protected:
  Factory() = default;
  Factory(const Factory &) = delete;
  Factory(Factory &&) = default;
};

} // namespace context
} // namespace fluidml

#endif
