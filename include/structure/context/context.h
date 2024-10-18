#ifndef CPU_TRANSFORMERS_STRUCTURE_CONTEXT_CONTEXT_H_
#define CPU_TRANSFORMERS_STRUCTURE_CONTEXT_CONTEXT_H_

#include "evaluation/fwd.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "structure/context/attr.h"
#include "structure/context/factory.h"
#include "structure/context/fwd.h"
#include "worker/fwd.h"
#include <memory>

namespace cpu_transformers {
namespace context {

class Context : public std::shared_ptr<ContextImpl> {
public:
  Context();
  Context(const Context &context) = default;
  Context(Context &&context) = default;
  Context(const std::shared_ptr<ContextImpl> &context_impl);
  Context(std::shared_ptr<ContextImpl> &&context_impl);
  Context &operator=(const Context &context) = default;
  Context &operator=(Context &&context) = default;
  Context &operator=(const std::shared_ptr<ContextImpl> &context_impl);
  Context &operator=(std::shared_ptr<ContextImpl> &&context_impl);
  virtual ~Context() = default;
  std::unique_ptr<evaluation::DynamicProgrammingTable>
  MakeDynamicProgrammingTable();
  std::unique_ptr<worker::GeneralBuilder>
  MakePlainGeneralBuilder(std::string &&function_name);
  std::unique_ptr<worker::GeneralBuilder>
  MakeDPGeneralBuilder(std::string &&function_name);
  std::unique_ptr<worker::KernelBuilder>
  MakeKernelBuilder(std::string &&function_name);
  std::unique_ptr<worker::Planner> MakePlainLinearPlanner();
  std::unique_ptr<worker::Planner> MakePlainGreedyPlanner();
  std::unique_ptr<worker::Planner> MakeDPGreedyPlanner();
  std::unique_ptr<worker::Lower> MakeLower();
  std::unique_ptr<worker::Runner> MakeRunner();
  friend std::ostream &operator<<(std::ostream &os, Context &context);
};

class ContextImpl {
public:
  virtual ~ContextImpl() = default;
  static std::unique_ptr<ContextImpl> Make();
  mlir::MLIRContext &GetMLIRContext();
  mlir::ModuleOp GetModule();
  FuncAttr &GetFuncAttr();
  Factory &GetFactory();
  void SetModule(mlir::OwningOpRef<mlir::ModuleOp> &&module);
  void SetFuncAttr(FuncAttr &&func_attr);
  std::unique_ptr<mlir::ExecutionEngine> MakeExecutionEngine();
  std::string ExportHeaderFile();
  friend std::ostream &operator<<(std::ostream &os, ContextImpl &context);

protected:
  ContextImpl();
  ContextImpl(const ContextImpl &context) = delete;
  ContextImpl(ContextImpl &&context) = delete;
  mlir::MLIRContext mlir_context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::optional<FuncAttr> func_attr_opt_;
  std::unique_ptr<Factory> factory_;
};

} // namespace context
} // namespace cpu_transformers

#endif