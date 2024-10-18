#ifndef CPU_TRANSFORMERS_EVALUATION_EVAL_H_
#define CPU_TRANSFORMERS_EVALUATION_EVAL_H_

#include "evaluation/fwd.h"
#include "nlohmann/json_fwd.hpp"
#include "structure/kernel/generator/generator.h"
#include "structure/tensor/meta.h"
#include "worker/fwd.h"
#include <cstddef>
#include <unordered_map>
#include <vector>

namespace ns {

void to_json(nlohmann::json &j,
             const cpu_transformers::evaluation::KernelEval &eval);

} // namespace ns

namespace cpu_transformers {
namespace evaluation {

class KernelEval {
public:
  KernelEval(size_t epoch);
  KernelEval(const KernelEval &) = delete;
  KernelEval(KernelEval &&) = default;
  virtual ~KernelEval() = default;
  virtual const kernel::KernelGenerator &GetKernelGenerator() const = 0;
  virtual kernel::KernelGenerator &GetKernelGenerator() = 0;
  virtual size_t GetShortestTimeCost() = 0;
  virtual nlohmann::json ToJson() const = 0;

protected:
  const size_t epoch;
};

class SingleInputKernelEval : public KernelEval {
public:
  class Key;
  struct KeyHash;

  class Key {
  public:
    Key(const std::vector<size_t> &input_shape,
        const std::vector<size_t> &output_shape);
    Key(std::vector<size_t> &&input_shape, std::vector<size_t> &&output_shape);
    Key(const Key &) = default;
    Key(Key &&) = default;
    ~Key() = default;
    bool operator==(const Key &rhs) const;
    friend class SingleInputKernelEval;
    friend struct SingleInputKernelEval::KeyHash;
    friend std::ostream &operator<<(std::ostream &os, const Key &key);

  private:
    const std::vector<size_t> input_shape_;
    const std::vector<size_t> output_shape_;
  };

  struct KeyHash {
    size_t operator()(const Key &key) const;
  };

  SingleInputKernelEval(size_t epoch);
  SingleInputKernelEval(const SingleInputKernelEval &) = delete;
  SingleInputKernelEval(SingleInputKernelEval &&) = default;
  virtual ~SingleInputKernelEval() = default;
  const kernel::SingleInputKernelGenerator &
  GetKernelGenerator() const override = 0;
  kernel::SingleInputKernelGenerator &GetKernelGenerator() override = 0;
  size_t GetTimeCost(const std::vector<size_t> &input_layout,
                     const std::vector<size_t> &output_layout);
  size_t GetShortestTimeCost() override;
  const Meta &GetInputMeta() const;
  const Meta &GetOutputMeta() const;
  nlohmann::json ToJson() const override;

protected:
  virtual void runKernel(worker::KernelBuilder &builer,
                         const std::vector<size_t> &input_layout,
                         const std::vector<size_t> &output_layout) const = 0;
  std::unordered_map<Key, size_t, KeyHash> time_costs_;
};

class SingleInputWithoutBufferKernelEval : public SingleInputKernelEval {
public:
  SingleInputWithoutBufferKernelEval(
      std::shared_ptr<kernel::SingleInputWithoutBufferKernelGenerator>
          &&generator,
      size_t epoch = 1);
  SingleInputWithoutBufferKernelEval(
      const SingleInputWithoutBufferKernelEval &) = delete;
  SingleInputWithoutBufferKernelEval(SingleInputWithoutBufferKernelEval &&) =
      default;
  virtual ~SingleInputWithoutBufferKernelEval() = default;
  const kernel::SingleInputWithoutBufferKernelGenerator &
  GetKernelGenerator() const override;
  kernel::SingleInputWithoutBufferKernelGenerator &
  GetKernelGenerator() override;

private:
  void runKernel(worker::KernelBuilder &builer,
                 const std::vector<size_t> &input_layout,
                 const std::vector<size_t> &output_layout) const override;
  std::shared_ptr<kernel::SingleInputWithoutBufferKernelGenerator> generator_;
};

class SingleInputWithBufferKernelEval : public SingleInputKernelEval {
public:
  SingleInputWithBufferKernelEval(
      std::shared_ptr<kernel::SingleInputWithBufferKernelGenerator> &&generator,
      size_t buffer_size, size_t epoch = 1);
  SingleInputWithBufferKernelEval(const SingleInputWithBufferKernelEval &) =
      delete;
  SingleInputWithBufferKernelEval(SingleInputWithBufferKernelEval &&) = default;
  virtual ~SingleInputWithBufferKernelEval() = default;
  const kernel::SingleInputWithBufferKernelGenerator &
  GetKernelGenerator() const override;
  kernel::SingleInputWithBufferKernelGenerator &GetKernelGenerator() override;

private:
  void runKernel(worker::KernelBuilder &builer,
                 const std::vector<size_t> &input_layout,
                 const std::vector<size_t> &output_layout) const override;
  std::shared_ptr<kernel::SingleInputWithBufferKernelGenerator> generator_;
  const size_t buffer_size_;
};

class DoubleInputsKernelEval : public KernelEval {
public:
  class Key;
  struct KeyHash;

  class Key {
  public:
    Key(const std::vector<size_t> &lhs_layout,
        const std::vector<size_t> &rhs_layout,
        const std::vector<size_t> &output_layout);
    Key(std::vector<size_t> &&lhs_layout, std::vector<size_t> &&rhs_layout,
        std::vector<size_t> &&output_layout);
    Key(const Key &) = default;
    Key(Key &&) = default;
    ~Key() = default;
    bool operator==(const Key &rhs) const;
    friend class DoubleInputsKernelEval;
    friend struct DoubleInputsKernelEval::KeyHash;
    friend std::ostream &operator<<(std::ostream &os, const Key &key);

  private:
    const std::vector<size_t> lhs_shape_;
    const std::vector<size_t> rhs_shape_;
    const std::vector<size_t> output_shape_;
  };

  struct KeyHash {
    size_t operator()(const Key &key) const;
  };

  DoubleInputsKernelEval(size_t epoch);
  DoubleInputsKernelEval(const DoubleInputsKernelEval &) = delete;
  DoubleInputsKernelEval(DoubleInputsKernelEval &&) = default;
  virtual ~DoubleInputsKernelEval() = default;
  const kernel::DoubleInputsKernelGenerator &
  GetKernelGenerator() const override = 0;
  kernel::DoubleInputsKernelGenerator &GetKernelGenerator() override = 0;
  size_t GetTimeCost(const std::vector<size_t> &lhs_layout,
                     const std::vector<size_t> &rhs_layout,
                     const std::vector<size_t> &output_layout);
  size_t GetShortestTimeCost() override;
  const Meta &GetLhsMeta() const;
  const Meta &GetRhsMeta() const;
  const Meta &GetOutputMeta() const;
  nlohmann::json ToJson() const override;

protected:
  virtual void runKernel(worker::KernelBuilder &builer,
                         const std::vector<size_t> &lhs_layout,
                         const std::vector<size_t> &rhs_layout,
                         const std::vector<size_t> &output_layout) const = 0;
  std::unordered_map<Key, size_t, KeyHash> time_costs_;
};

class DoubleInputsWithoutBufferKernelEval : public DoubleInputsKernelEval {
public:
  DoubleInputsWithoutBufferKernelEval(
      std::shared_ptr<kernel::DoubleInputsWithoutBufferKernelGenerator>
          &&generator,
      size_t epoch = 1);
  DoubleInputsWithoutBufferKernelEval(
      const DoubleInputsWithoutBufferKernelEval &) = delete;
  DoubleInputsWithoutBufferKernelEval(DoubleInputsWithoutBufferKernelEval &&) =
      default;
  virtual ~DoubleInputsWithoutBufferKernelEval() = default;
  const kernel::DoubleInputsWithoutBufferKernelGenerator &
  GetKernelGenerator() const override;
  kernel::DoubleInputsWithoutBufferKernelGenerator &
  GetKernelGenerator() override;

private:
  void runKernel(worker::KernelBuilder &builer,
                 const std::vector<size_t> &lhs_layout,
                 const std::vector<size_t> &rhs_layout,
                 const std::vector<size_t> &output_layout) const override;
  std::shared_ptr<kernel::DoubleInputsWithoutBufferKernelGenerator> generator_;
};

// TODO: based on the current implementation, there is still no
// `DoubleInputsWithBufferKernelEval` class yet, so it's not included here

} // namespace evaluation
} // namespace cpu_transformers

#endif
