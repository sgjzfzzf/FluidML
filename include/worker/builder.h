#ifndef CPU_TRANSFORMERS_WORKER_BUILDERS_H_
#define CPU_TRANSFORMERS_WORKER_BUILDERS_H_

#include "structure/context/context.h"
#include "structure/flow/sequence.h"
#include "structure/kernel/kernel.h"
#include "structure/memory/index.h"
#include "structure/tensor/meta.h"
#include "worker/fwd.h"
#include <memory>

namespace cpu_transformers {
namespace worker {

class Builder {
public:
  virtual ~Builder() = default;

protected:
  Builder() = default;
  Builder(const Builder &) = delete;
  Builder(Builder &&) = default;
};

class GeneralBuilder : public Builder {
public:
  virtual ~GeneralBuilder() = default;
  virtual void Run(const flow::Sequence &sequence,
                   const memory::Index &index) = 0;
  static std::unique_ptr<GeneralBuilder> Make(std::string &&function_name,
                                              context::Context &&context);

protected:
  GeneralBuilder() = default;
  GeneralBuilder(const GeneralBuilder &) = delete;
  GeneralBuilder(GeneralBuilder &&) = default;
};

class KernelBuilder : public Builder {
public:
  virtual ~KernelBuilder() = default;
  virtual void RunOnSingleInputWithoutBuffer(
      const kernel::SingleInputWithoutBufferKernel &kernel,
      const Meta &input_meta, const Meta &output_meta) = 0;
  virtual void RunOnSingleInputWithoutBuffer(
      const kernel::SingleInputWithoutBufferKernel &kernel,
      const Meta &input_meta, const std::vector<size_t> &input_layout,
      const Meta &output_meta, const std::vector<size_t> &output_layout) = 0;
  virtual void
  RunOnSingleInputWithBuffer(const kernel::SingleInputWithBufferKernel &kernel,
                             const Meta &input_meta, const Meta &output_meta,
                             size_t buffer_size) = 0;
  virtual void RunOnSingleInputWithBuffer(
      const kernel::SingleInputWithBufferKernel &kernel, const Meta &input_meta,
      const std::vector<size_t> &input_layout, const Meta &output_meta,
      const std::vector<size_t> &output_layout, size_t buffer_size) = 0;
  virtual void RunOnDoubleInputsWithoutBuffer(
      const kernel::DoubleInputsWithoutBufferKernel &kernel,
      const Meta &lhs_meta, const Meta &rhs_meta, const Meta &output_meta) = 0;
  virtual void RunOnDoubleInputsWithoutBuffer(
      const kernel::DoubleInputsWithoutBufferKernel &kernel,
      const Meta &lhs_meta, const std::vector<size_t> &lhs_layout,
      const Meta &rhs_meta, const std::vector<size_t> &rhs_layout,
      const Meta &output_meta, const std::vector<size_t> &output_layout) = 0;
  static constexpr char kInputKey[] = "input";
  static constexpr char kLhsKey[] = "lhs";
  static constexpr char kRhsKey[] = "rhs";
  static constexpr char kOutputKey[] = "output";
  static std::unique_ptr<KernelBuilder> Make(std::string &&function_name,
                                             context::Context &&context);

protected:
  KernelBuilder() = default;
  KernelBuilder(const KernelBuilder &) = delete;
  KernelBuilder(KernelBuilder &&) = default;
};

} // namespace worker
} // namespace cpu_transformers

#endif
