#ifndef CPU_TRANSFORMERS_WORKER_UTILS_H_
#define CPU_TRANSFORMERS_WORKER_UTILS_H_

#include "structure/flow/node.h"
#include "structure/kernel/generator/generator.h"
#include <memory>

namespace cpu_transformers {
namespace worker {

std::string GetBufferName(std::string_view name);

std::unique_ptr<kernel::Kernel> SelectKernel(const flow::Node *node);

std::unique_ptr<kernel::KernelGenerator>
SelectKernelGenerator(const flow::Node *node);

} // namespace worker
} // namespace cpu_transformers

#endif
