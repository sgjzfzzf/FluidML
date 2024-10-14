#include "worker/scheduler.h"
#include "mlir/IR/Builders.h"
#include "structure/flow/node.h"
#include "structure/kernel/kernel.h"
#include "worker/utils.h"
#include <memory>
#include <string>
#include <unordered_map>
#ifdef DEBUG
#include <cassert>
#endif

namespace cpu_transformers {
namespace worker {

class SchedulerImpl : public Scheduler {
public:
  SchedulerImpl() = default;
  SchedulerImpl(const SchedulerImpl &scheduler) = delete;
  SchedulerImpl(SchedulerImpl &&scheduler) = default;
  virtual ~SchedulerImpl() = default;
  void Run(mlir::OpBuilder &builder, const flow::Sequence &sequence,
           std::unordered_map<std::string, mlir::Value> &symbol_table) override;
};

std::unique_ptr<Scheduler> Scheduler::Make() {
  return std::make_unique<SchedulerImpl>();
}

void SchedulerImpl::Run(
    mlir::OpBuilder &builder, const flow::Sequence &sequence,
    std::unordered_map<std::string, mlir::Value> &symbol_table) {
  const std::vector<std::shared_ptr<flow::Node>> &nodes = sequence.GetNodes();
  for (std::shared_ptr<flow::Node> node : nodes) {
    std::shared_ptr<kernel::Kernel> k = SelectKernel(node.get());
#ifdef DEBUG
    assert(k != nullptr);
#endif
    if (std::shared_ptr<flow::SingleInputWithoutBufferNode> ptr =
            std::dynamic_pointer_cast<flow::SingleInputWithoutBufferNode>(
                node)) {
      std::shared_ptr<kernel::SingleInputWithoutBufferKernel> kernel =
          std::dynamic_pointer_cast<kernel::SingleInputWithoutBufferKernel>(k);
#ifdef DEBUG
      assert(kernel != nullptr);
#endif
      const std::string &input_name = ptr->GetInputAsString();
      const std::string &output_name = ptr->GetOutputAsString();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      kernel->Run(builder, input, output);
    } else if (std::shared_ptr<flow::SingleInputWithBufferNode> ptr =
                   std::dynamic_pointer_cast<flow::SingleInputWithBufferNode>(
                       node)) {
      std::shared_ptr<kernel::SingleInputWithBufferKernel> kernel =
          std::dynamic_pointer_cast<kernel::SingleInputWithBufferKernel>(k);
#ifdef DEBUG
      assert(kernel != nullptr);
#endif
      const std::string &input_name = ptr->GetInputAsString();
      const std::string &output_name = ptr->GetOutputAsString();
      const std::string &buffer_name = ptr->GetName();
      mlir::Value &input = symbol_table.at(input_name);
      mlir::Value &output = symbol_table.at(output_name);
      mlir::Value &buffer = symbol_table.at(buffer_name);
      kernel->Run(builder, input, output, buffer);
    } else if (std::shared_ptr<flow::DoubleInputsWithoutBufferNode> ptr =
                   std::dynamic_pointer_cast<
                       flow::DoubleInputsWithoutBufferNode>(node)) {
      std::shared_ptr<kernel::DoubleInputsWithoutBufferKernel> kernel =
          std::dynamic_pointer_cast<kernel::DoubleInputsWithoutBufferKernel>(k);
#ifdef DEBUG
      assert(kernel != nullptr);
#endif
      const std::string &lhs_name = ptr->GetLhsAsString();
      const std::string &rhs_name = ptr->GetRhsAsString();
      const std::string &output_name = ptr->GetOutputAsString();
      mlir::Value &lhs = symbol_table.at(lhs_name);
      mlir::Value &rhs = symbol_table.at(rhs_name);
      mlir::Value &output = symbol_table.at(output_name);
      kernel->Run(builder, lhs, rhs, output);
    } else if (std::shared_ptr<flow::DoubleInputsWithBufferNode> ptr =
                   std::dynamic_pointer_cast<flow::DoubleInputsWithBufferNode>(
                       node)) {
      std::shared_ptr<kernel::DoubleInputsWithBufferKernel> kernel =
          std::dynamic_pointer_cast<kernel::DoubleInputsWithBufferKernel>(k);
#ifdef DEBUG
      assert(kernel != nullptr);
#endif
      const std::string &lhs_name = ptr->GetLhsAsString();
      const std::string &rhs_name = ptr->GetRhsAsString();
      const std::string &output_name = ptr->GetOutputAsString();
      const std::string &buffer_name = ptr->GetName();
      mlir::Value &lhs = symbol_table.at(lhs_name);
      mlir::Value &rhs = symbol_table.at(rhs_name);
      mlir::Value &output = symbol_table.at(output_name);
      mlir::Value &buffer = symbol_table.at(buffer_name);
      kernel->Run(builder, lhs, rhs, output, buffer);
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
  }
}

} // namespace worker
} // namespace cpu_transformers
