#ifndef CPU_TRANSFORMERS_WORKER_UTILS_H_
#define CPU_TRANSFORMERS_WORKER_UTILS_H_

#include "structure/flow/node.h"
#include "structure/kernel/kernel/kernel.h"
#include <memory>

namespace cpu_transformers {
namespace worker {

std::shared_ptr<kernel::Kernel> SelectKernel(const flow::Node *node);

}
} // namespace cpu_transformers

#endif
