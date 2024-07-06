#ifndef CPU_TRANSFORMERS_WORKER_BUILDERS_H_
#define CPU_TRANSFORMERS_WORKER_BUILDERS_H_

#include "structure/context/context.h"
#include "structure/flow/sequence.h"
#include "structure/kernel/kernel.h"
#include "structure/memory/index.h"
#include "structure/tensor/meta.h"
#include "worker/worker.h"
#include <memory>

namespace cpu_transformers {
namespace worker {

class Builder {
public:
  Builder() = default;
  Builder(const Builder &) = delete;
  Builder(Builder &&) = default;
};

class GeneralBuilder : public Builder {
public:
  GeneralBuilder(std::string &&function_name = "main",
                 std::shared_ptr<context::Context> context = nullptr);
  GeneralBuilder(const GeneralBuilder &builder) = delete;
  GeneralBuilder(GeneralBuilder &&builder) = default;
  virtual ~GeneralBuilder();
  void Run(const flow::Sequence &sequence, const memory::Index &index);

private:
  const std::string function_name_;
  std::shared_ptr<context::Context> context_;
  std::unique_ptr<Scheduler> scheduler_;
};

class KernelBuilder : public Builder {
public:
  KernelBuilder(std::string &&function_name = "main",
                std::shared_ptr<context::Context> context = nullptr);
  KernelBuilder(const KernelBuilder &) = delete;
  KernelBuilder(KernelBuilder &&) = default;
  void RunOnSingleInputWithoutBuffer(
      const kernel::SingleInputWithoutBufferKernel &kernel,
      const Meta &input_meta, const Meta &output_meta);
  void RunOnSingleInputWithoutBuffer(
      const kernel::SingleInputWithoutBufferKernel &kernel,
      const Meta &input_meta, const std::vector<size_t> &input_layout,
      const Meta &output_meta, const std::vector<size_t> &output_layout);
  void
  RunOnSingleInputWithBuffer(const kernel::SingleInputWithBufferKernel &kernel,
                             const Meta &input_meta, const Meta &output_meta,
                             size_t buffer_size);
  void RunOnSingleInputWithBuffer(
      const kernel::SingleInputWithBufferKernel &kernel, const Meta &input_meta,
      const std::vector<size_t> &input_layout, const Meta &output_meta,
      const std::vector<size_t> &output_layout, size_t buffer_size);
  void RunOnDoubleInputsWithoutBuffer(
      const kernel::DoubleInputsWithoutBufferKernel &kernel,
      const Meta &lhs_meta, const Meta &rhs_meta, const Meta &output_meta);
  void RunOnDoubleInputsWithoutBuffer(
      const kernel::DoubleInputsWithoutBufferKernel &kernel,
      const Meta &lhs_meta, const std::vector<size_t> &lhs_layout,
      const Meta &rhs_meta, const std::vector<size_t> &rhs_layout,
      const Meta &output_meta, const std::vector<size_t> &output_layout);
  static constexpr char kInputKey[] = "input";
  static constexpr char kLhsKey[] = "lhs";
  static constexpr char kRhsKey[] = "rhs";
  static constexpr char kOutputKey[] = "output";

private:
  std::string function_name_;
  std::shared_ptr<context::Context> context_;
};

} // namespace worker
} // namespace cpu_transformers

#endif
