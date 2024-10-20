#ifndef CPU_TRANSFORMERS_WORKER_EXECUTOR_H_
#define CPU_TRANSFORMERS_WORKER_EXECUTOR_H_

#ifdef BUILD_PYTHON
#include "pybind11/numpy.h"
#endif
#include <fstream>
#include <memory>
#include <optional>
#include <unordered_map>

namespace cpu_transformers {
namespace worker {

class Executor {
public:
  virtual ~Executor() = default;
  void Compile(std::string_view inpupt,
               std::optional<std::string_view> mlir = std::nullopt,
               std::optional<std::string_view> llvm = std::nullopt);
  virtual void Compile(std::istream &input, std::ofstream *mlir = nullptr,
                       std::ofstream *llvm = nullptr) = 0;
  virtual size_t
  Invoke(const std::unordered_map<std::string, void *> &args) = 0;
#ifdef BUILD_PYTHON
  virtual size_t
  Invoke(const std::unordered_map<std::string, pybind11::array> &args) = 0;
#endif
  static std::unique_ptr<Executor> MakePlainLinear(std::string &&name,
                                                   size_t epoch = 1);
  static std::unique_ptr<Executor> MakePlainGreedy(std::string &&name,
                                                   size_t epoch = 1);
  static std::unique_ptr<Executor> MakeDPGreedy(std::string &&name,
                                                size_t epoch = 1);

protected:
  Executor() = default;
  Executor(const Executor &) = delete;
  Executor(Executor &&) = default;
};

} // namespace worker
} // namespace cpu_transformers

#endif
