#include "worker/runner.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "structure/context/attr.h"
#include "structure/context/context.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <unordered_map>
#ifdef DEBUG
#include <cassert>
#endif

namespace {
extern "C" struct MemRefDescriptor {
  void *base;
  void *alloc;
  int64_t offset;
  // Note: "size" includes both size and stride information
  int64_t size[];
};
static_assert(std::is_trivial_v<MemRefDescriptor>);
} // namespace

namespace cpu_transformers {
namespace worker {

class RunnerImpl : public Runner {
public:
  RunnerImpl(context::Context &&context);
  RunnerImpl(const RunnerImpl &runner) = delete;
  RunnerImpl(RunnerImpl &&runner) = default;
  virtual ~RunnerImpl() = default;
  size_t Run(const std::unordered_map<std::string, void *> &args,
             size_t epoch) override;
#ifdef BUILD_PYTHON
  size_t Run(const std::unordered_map<std::string, pybind11::array> &args,
             size_t epoch) override;
#endif

private:
  context::Context context_;
};

std::unique_ptr<Runner> Runner::Make(context::Context &&context) {
  return std::make_unique<RunnerImpl>(std::move(context));
}

RunnerImpl::RunnerImpl(context::Context &&context) : context_(context) {}

size_t RunnerImpl::Run(const std::unordered_map<std::string, void *> &args,
                       size_t epoch) {
  mlir::MLIRContext &mlir_context = context_->GetMLIRContext();
  std::unique_ptr<mlir::ExecutionEngine> engine =
      context_->MakeExecutionEngine();
  mlir::ModuleOp module = context_->GetModule();
  const context::FuncAttr &func_attr = context_->GetFuncAttr();
  std::string func_name = func_attr.GetName();
  const std::vector<context::ArgumentAttr> &arguments =
      func_attr.GetArguments();
  llvm::SmallVector<void *> descs;
  for (const context::ArgumentAttr &argument : arguments) {
    std::string name = argument.GetName();
    mlir::MemRefType type = argument.GetMemRefType();
    void *ref_base_ptr = args.at(name);
#ifdef DEBUG
    assert(ref_base_ptr != nullptr);
#endif
    llvm::ArrayRef<int64_t> shape = type.getShape();
    const int64_t len = shape.size();
    int64_t mem = type.getNumElements() * type.getElementTypeBitWidth() / 8;
    MemRefDescriptor *desc = reinterpret_cast<MemRefDescriptor *>(
        std::malloc(sizeof(MemRefDescriptor) + 2 * len * sizeof(int64_t)));
    desc->base = ref_base_ptr;
    desc->alloc = ref_base_ptr;
    desc->offset = 0;
    MemRefDescriptor **desc_ptr = reinterpret_cast<MemRefDescriptor **>(
        std::malloc(sizeof(MemRefDescriptor *)));
    *desc_ptr = desc;
    descs.push_back(desc_ptr);
  }
  const int64_t buffer_size = func_attr.GetBuffer();
  std::vector<uint8_t> buffer(buffer_size, 0);
  void *base_ptr = buffer.data();
  MemRefDescriptor *desc = reinterpret_cast<MemRefDescriptor *>(
      std::malloc(sizeof(MemRefDescriptor) + 2 * sizeof(int64_t)));
  desc->base = base_ptr;
  desc->alloc = base_ptr;
  desc->offset = 0;
  desc->size[0] = buffer_size;
  desc->size[1] = 1;
  MemRefDescriptor **desc_ptr = reinterpret_cast<MemRefDescriptor **>(
      std::malloc(sizeof(MemRefDescriptor *)));
  *desc_ptr = desc;
  descs.push_back(desc_ptr);
  std::string mlir_func_name = "_mlir_ciface_" + func_name;
  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < epoch; ++i) {
    llvm::Error error = engine->invokePacked(mlir_func_name, descs);
#ifdef DEBUG
    assert(!error);
#endif
  }
  std::chrono::high_resolution_clock::time_point end =
      std::chrono::high_resolution_clock::now();
  auto duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  for (void *desc : descs) {
    MemRefDescriptor *p = *reinterpret_cast<MemRefDescriptor **>(desc);
    std::free(p);
    std::free(desc);
  }
  return duration_ns.count() / epoch;
}

#ifdef BUILD_PYTHON
size_t
RunnerImpl::Run(const std::unordered_map<std::string, pybind11::array> &args,
                size_t epoch) {
  std::unordered_map<std::string, void *> ptrs;
  for (const auto &input : args) {
    const std::string &name = input.first;
    const pybind11::array &array = input.second;
    pybind11::buffer_info info = array.request();
    void *ptr = info.ptr;
    ptrs.insert({name, ptr});
  }
  return Run(ptrs, epoch);
}
#endif

} // namespace worker
} // namespace cpu_transformers
