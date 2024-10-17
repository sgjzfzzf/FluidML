#include "evaluation/eval.h"
#include "fmt/ranges.h"
#include "nlohmann/json.hpp"
#include "structure/kernel/generator/generator.h"
#include "structure/kernel/kernel/kernel.h"
#include "utils/hash.h"
#include "utils/utils.h"
#include "worker/builder.h"
#ifndef DP_DEBUG
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "worker/lower.h"
#include "worker/runner.h"
#endif

namespace ns {

void to_json(nlohmann::json &j,
             const cpu_transformers::evaluation::KernelEval &eval) {
  j = eval.ToJson();
}

} // namespace ns

namespace cpu_transformers {
namespace evaluation {

KernelEval::KernelEval(size_t epoch) : epoch(epoch) {}

SingleInputKernelEval::Key::Key(const std::vector<size_t> &input_shape,
                                const std::vector<size_t> &output_shape)
    : input_shape_(input_shape), output_shape_(output_shape) {}

SingleInputKernelEval::Key::Key(std::vector<size_t> &&input_shape,
                                std::vector<size_t> &&output_shape)
    : input_shape_(std::move(input_shape)),
      output_shape_(std::move(output_shape)) {}

bool SingleInputKernelEval::Key::operator==(const Key &rhs) const {
  return input_shape_ == rhs.input_shape_ && output_shape_ == rhs.output_shape_;
}

std::ostream &operator<<(std::ostream &os,
                         const SingleInputKernelEval::Key &key) {
  os << fmt::format("{},{}", key.input_shape_, key.output_shape_);
  return os;
}

size_t SingleInputKernelEval::KeyHash::operator()(const Key &key) const {
  size_t hash = 0;
  std::hash<int64_t> hasher;
  for (int64_t i : key.input_shape_) {
    hash ^= hasher(i) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  for (int64_t i : key.output_shape_) {
    hash ^= hasher(i) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  return hash;
};

size_t
SingleInputKernelEval::GetTimeCost(const std::vector<size_t> &input_layout,
                                   const std::vector<size_t> &output_layout) {
  // Add a cache to save time on evaluate the same kernel.
  auto it = time_costs_.find({input_layout, output_layout});
  if (it != time_costs_.end()) {
    return it->second;
  }
  size_t time_cost;
#ifdef DP_DEBUG
  // TODO: this return statement is a placeholder, to accelerate the execution
  // during development
  time_cost = 1;
#else
  kernel::SingleInputKernelGenerator &generator = GetKernelGenerator();
  std::string kernel_name = generator.GetKernelName();
  mlir::MLIRContext mlir_context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlir_context));
  context::Context context;
  context->SetModule(std::move(module));
  std::unique_ptr<worker::KernelBuilder> builder =
      context.MakeKernelBuilder(kernel_name.c_str());
  runKernel(*builder, input_layout, output_layout);
  std::unique_ptr<worker::Lower> lower = context.MakeLower();
  lower->Run();
  std::unique_ptr<worker::Runner> runner = context.MakeRunner();
  const Meta &input_meta = GetInputMeta(), output_meta = GetOutputMeta();
  std::vector<uint8_t> input_buffer = utils::FillBuffer(input_meta),
                       output_buffer = utils::FillBuffer(output_meta);
  time_cost =
      runner->Run({{worker::KernelBuilder::kInputKey, input_buffer.data()},
                   {worker::KernelBuilder::kOutputKey, output_buffer.data()}});
#endif
  time_costs_.insert_or_assign({input_layout, output_layout}, time_cost);
  return time_cost;
}

size_t SingleInputKernelEval::GetShortestTimeCost() {
  size_t shortest_time_cost = std::numeric_limits<size_t>::max();
  const Meta &input_meta = GetInputMeta(), output_meta = GetOutputMeta();
  const std::vector<int64_t> &input_shape = input_meta.GetShape(),
                             &output_shape = output_meta.GetShape();
  const size_t input_shape_len = input_shape.size(),
               output_shape_len = output_shape.size();
  std::vector<std::vector<size_t>> input_layouts =
                                       utils::GenAllOrders(input_shape_len),
                                   output_layouts =
                                       utils::GenAllOrders(output_shape_len);
  for (const std::vector<size_t> &input_layout : input_layouts) {
    for (const std::vector<size_t> &output_layout : output_layouts) {
      const size_t time_cost = GetTimeCost(input_layout, output_layout);
      if (time_cost < shortest_time_cost) {
        shortest_time_cost = time_cost;
      }
    }
  }
  return shortest_time_cost;
}

const Meta &SingleInputKernelEval::GetInputMeta() const {
  const kernel::SingleInputKernelGenerator &generator = GetKernelGenerator();
  const Meta &input_meta = generator.GetInputMeta();
  return input_meta;
}

const Meta &SingleInputKernelEval::GetOutputMeta() const {
  const kernel::SingleInputKernelGenerator &generator = GetKernelGenerator();
  const Meta &output_meta = generator.GetOutputMeta();
  return output_meta;
}

nlohmann::json SingleInputKernelEval::ToJson() const {
  nlohmann::json json;
  for (const auto &[key, value] : time_costs_) {
    nlohmann::json elem = {
        {"input_shape", key.input_shape_},
        {"output_shape", key.output_shape_},
        {"time_cost", value},
    };
    json.push_back(std::move(elem));
  }
  return json;
}

SingleInputWithoutBufferKernelEval::SingleInputWithoutBufferKernelEval(
    std::shared_ptr<kernel::SingleInputWithoutBufferKernelGenerator>
        &&generator)
    : generator_(std::move(generator)) {}

const kernel::SingleInputWithoutBufferKernelGenerator &
SingleInputWithoutBufferKernelEval::GetKernelGenerator() const {
#ifdef DEBUG
  assert(generator_ != nullptr);
#endif
  return *generator_;
}

kernel::SingleInputWithoutBufferKernelGenerator &
SingleInputWithoutBufferKernelEval::GetKernelGenerator() {
#ifdef DEBUG
  assert(generator_ != nullptr);
#endif
  return *generator_;
}

void SingleInputWithoutBufferKernelEval::runKernel(
    worker::KernelBuilder &builer, const std::vector<size_t> &input_layout,
    const std::vector<size_t> &output_layout) const {
  std::shared_ptr<kernel::SingleInputWithoutBufferKernel> kernel =
      generator_->YieldSingleInputWithoutBufferKernel(input_layout,
                                                      output_layout);
  const Meta &input_meta = GetInputMeta(), output_meta = GetOutputMeta();
  builer.RunOnSingleInputWithoutBuffer(*kernel, input_meta, input_layout,
                                       output_meta, output_layout);
}

SingleInputWithBufferKernelEval::SingleInputWithBufferKernelEval(
    std::shared_ptr<kernel::SingleInputWithBufferKernelGenerator> &&generator,
    size_t buffer_size)
    : generator_(std::move(generator)), buffer_size_(buffer_size) {}

const kernel::SingleInputWithBufferKernelGenerator &
SingleInputWithBufferKernelEval::GetKernelGenerator() const {
#ifdef DEBUG
  assert(generator_ != nullptr);
#endif
  return *generator_;
}

kernel::SingleInputWithBufferKernelGenerator &
SingleInputWithBufferKernelEval::GetKernelGenerator() {
#ifdef DEBUG
  assert(generator_ != nullptr);
#endif
  return *generator_;
}

void SingleInputWithBufferKernelEval::runKernel(
    worker::KernelBuilder &builer, const std::vector<size_t> &input_layout,
    const std::vector<size_t> &output_layout) const {
  std::shared_ptr<kernel::SingleInputWithBufferKernel> kernel =
      generator_->YieldSingleInputWithBufferKernel(input_layout, output_layout);
  const Meta &input_meta = GetInputMeta(), output_meta = GetOutputMeta();
  builer.RunOnSingleInputWithBuffer(*kernel, input_meta, input_layout,
                                    output_meta, output_layout, buffer_size_);
}

DoubleInputsKernelEval::Key::Key(const std::vector<size_t> &lhs_shape,
                                 const std::vector<size_t> &rhs_shape,
                                 const std::vector<size_t> &output_shape)
    : lhs_shape_(lhs_shape), rhs_shape_(rhs_shape),
      output_shape_(output_shape) {}

DoubleInputsKernelEval::Key::Key(std::vector<size_t> &&lhs_shape,
                                 std::vector<size_t> &&rhs_shape,
                                 std::vector<size_t> &&output_shape)
    : lhs_shape_(std::move(lhs_shape)), rhs_shape_(std::move(rhs_shape)),
      output_shape_(std::move(output_shape)) {}

bool DoubleInputsKernelEval::Key::operator==(const Key &rhs) const {
  return lhs_shape_ == rhs.lhs_shape_ && rhs_shape_ == rhs.rhs_shape_ &&
         output_shape_ == rhs.output_shape_;
}

std::ostream &operator<<(std::ostream &os,
                         const DoubleInputsKernelEval::Key &key) {
  os << fmt::format("{},{},{}", key.lhs_shape_, key.rhs_shape_,
                    key.output_shape_);
  return os;
}

size_t DoubleInputsKernelEval::KeyHash::operator()(const Key &key) const {
  size_t hash = 0;
  std::hash<int64_t> hasher;
  for (int64_t i : key.lhs_shape_) {
    hash ^= hasher(i) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  for (int64_t i : key.rhs_shape_) {
    hash ^= hasher(i) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  for (int64_t i : key.output_shape_) {
    hash ^= hasher(i) + kHashSeed + (hash << 6) + (hash >> 2);
  }
  return hash;
};

size_t
DoubleInputsKernelEval::GetTimeCost(const std::vector<size_t> &lhs_layout,
                                    const std::vector<size_t> &rhs_layout,
                                    const std::vector<size_t> &output_layout) {
  size_t time_cost;
  kernel::DoubleInputsKernelGenerator &generator = GetKernelGenerator();
  std::string kernel_name = generator.GetKernelName();
#ifdef DP_DEBUG
  // TODO: this return statement is a placeholder, to accelerate the execution
  // during development
  time_cost = 1;
#else
  // Add a cache to save time on evaluate the same kernel.
  auto it = time_costs_.find({lhs_layout, rhs_layout, output_layout});
  if (it != time_costs_.end()) {
    return it->second;
  }
  mlir::MLIRContext mlir_context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlir_context));
  context::Context context;
  context->SetModule(std::move(module));
  std::unique_ptr<worker::KernelBuilder> builder =
      context.MakeKernelBuilder(kernel_name.c_str());
  runKernel(*builder, lhs_layout, rhs_layout, output_layout);
  std::unique_ptr<worker::Lower> lower = context.MakeLower();
  lower->Run();
  std::unique_ptr<worker::Runner> runner = context.MakeRunner();
  const Meta &lhs_meta = GetLhsMeta(), rhs_meta = GetRhsMeta(),
             output_meta = GetOutputMeta();
  std::vector<uint8_t> lhs_buffer = utils::FillBuffer(lhs_meta),
                       rhs_buffer = utils::FillBuffer(rhs_meta),
                       output_buffer = utils::FillBuffer(output_meta);
  time_cost =
      runner->Run({{worker::KernelBuilder::kLhsKey, lhs_buffer.data()},
                   {worker::KernelBuilder::kRhsKey, rhs_buffer.data()},
                   {worker::KernelBuilder::kOutputKey, output_buffer.data()}});
#endif
  time_costs_.insert_or_assign({lhs_layout, rhs_layout, output_layout},
                               time_cost);
  return time_cost;
}

size_t DoubleInputsKernelEval::GetShortestTimeCost() {
  size_t shortest_time_cost = std::numeric_limits<size_t>::max();
  const Meta &lhs_meta = GetLhsMeta(), rhs_meta = GetRhsMeta(),
             output_meta = GetOutputMeta();
  const std::vector<int64_t> &lhs_shape = lhs_meta.GetShape(),
                             &rhs_shape = rhs_meta.GetShape(),
                             &output_shape = output_meta.GetShape();
  const size_t lhs_shape_len = lhs_shape.size(),
               rhs_shape_len = rhs_shape.size(),
               output_shape_len = output_shape.size();
  std::vector<std::vector<size_t>> lhs_layouts =
                                       utils::GenAllOrders(lhs_shape_len),
                                   rhs_layouts =
                                       utils::GenAllOrders(rhs_shape_len),
                                   output_layouts =
                                       utils::GenAllOrders(output_shape_len);
  for (const std::vector<size_t> &lhs_layout : lhs_layouts) {
    for (const std::vector<size_t> &rhs_layout : rhs_layouts) {
      for (const std::vector<size_t> &output_layout : output_layouts) {
        const size_t time_cost =
            GetTimeCost(lhs_layout, rhs_layout, output_layout);
        if (time_cost < shortest_time_cost) {
          shortest_time_cost = time_cost;
        }
      }
    }
  }
  return shortest_time_cost;
}

const Meta &DoubleInputsKernelEval::GetLhsMeta() const {
  const kernel::DoubleInputsKernelGenerator &generator = GetKernelGenerator();
  const Meta &lhs_meta = generator.GetLhsMeta();
  return lhs_meta;
}

const Meta &DoubleInputsKernelEval::GetRhsMeta() const {
  const kernel::DoubleInputsKernelGenerator &generator = GetKernelGenerator();
  const Meta &rhs_meta = generator.GetRhsMeta();
  return rhs_meta;
}

const Meta &DoubleInputsKernelEval::GetOutputMeta() const {
  const kernel::DoubleInputsKernelGenerator &generator = GetKernelGenerator();
  const Meta &output_meta = generator.GetOutputMeta();
  return output_meta;
}

nlohmann::json DoubleInputsKernelEval::ToJson() const {
  nlohmann::json json;
  for (const auto &[key, value] : time_costs_) {
    nlohmann::json elem = {
        {"lhs_shape", key.lhs_shape_},
        {"rhs_shape", key.rhs_shape_},
        {"output_shape", key.output_shape_},
        {"time_cost", value},
    };
    json.push_back(std::move(elem));
  }
  return json;
}

DoubleInputsWithoutBufferKernelEval::DoubleInputsWithoutBufferKernelEval(
    std::shared_ptr<kernel::DoubleInputsWithoutBufferKernelGenerator>
        &&generator)
    : generator_(std::move(generator)) {}

const kernel::DoubleInputsWithoutBufferKernelGenerator &
DoubleInputsWithoutBufferKernelEval::GetKernelGenerator() const {
#ifdef DEBUG
  assert(generator_ != nullptr);
#endif
  return *generator_;
}

kernel::DoubleInputsWithoutBufferKernelGenerator &
DoubleInputsWithoutBufferKernelEval::GetKernelGenerator() {
#ifdef DEBUG
  assert(generator_ != nullptr);
#endif
  return *generator_;
}

void DoubleInputsWithoutBufferKernelEval::runKernel(
    worker::KernelBuilder &builer, const std::vector<size_t> &lhs_layout,
    const std::vector<size_t> &rhs_layout,
    const std::vector<size_t> &output_layout) const {
  std::shared_ptr<kernel::DoubleInputsWithoutBufferKernel> kernel =
      generator_->YieldDoubleInputsWithoutBufferKernel(lhs_layout, rhs_layout,
                                                       output_layout);
  const Meta &lhs_meta = GetLhsMeta(), rhs_meta = GetRhsMeta(),
             output_meta = GetOutputMeta();
  builer.RunOnDoubleInputsWithoutBuffer(*kernel, lhs_meta, lhs_layout, rhs_meta,
                                        rhs_layout, output_meta, output_layout);
}

} // namespace evaluation
} // namespace cpu_transformers
