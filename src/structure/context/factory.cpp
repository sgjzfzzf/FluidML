#include "structure/context/factory.h"
#include "worker/utils.h"
#include <unordered_set>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace context {

class FactoryImpl : public Factory {
public:
  FactoryImpl() = default;
  FactoryImpl(const FactoryImpl &) = delete;
  FactoryImpl(FactoryImpl &&) = default;
  virtual ~FactoryImpl() = default;
  std::shared_ptr<kernel::KernelGenerator>
  MakeKernelGenerator(const flow::Node &node) override;

private:
  struct KeyHash {
    size_t operator()(std::shared_ptr<kernel::KernelGenerator> generator) const;
  };

  struct KeyEqual {
    bool operator()(std::shared_ptr<kernel::KernelGenerator> lhs,
                    std::shared_ptr<kernel::KernelGenerator> rhs) const;
  };

  std::unordered_set<std::shared_ptr<kernel::KernelGenerator>, KeyHash,
                     KeyEqual>
      kernel_generator_set_;
};

std::unique_ptr<Factory> Factory::Make() {
  return std::make_unique<FactoryImpl>();
}

std::shared_ptr<kernel::KernelGenerator>
FactoryImpl::MakeKernelGenerator(const flow::Node &node) {
  std::shared_ptr<kernel::KernelGenerator> kernel_generator =
      worker::SelectKernelGenerator(&node);
  auto it = kernel_generator_set_.find(kernel_generator);
  if (it == kernel_generator_set_.end()) {
    kernel_generator_set_.insert(kernel_generator);
  } else {
    kernel_generator = *it;
  }
  return kernel_generator;
}

size_t FactoryImpl::KeyHash::operator()(
    std::shared_ptr<kernel::KernelGenerator> generator) const {
  return generator->GetHashCode();
}

bool FactoryImpl::KeyEqual::operator()(
    std::shared_ptr<kernel::KernelGenerator> lhs,
    std::shared_ptr<kernel::KernelGenerator> rhs) const {
  return lhs->Equals(*rhs);
}

} // namespace context
} // namespace cpu_transformers
