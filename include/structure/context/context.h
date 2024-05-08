#ifndef CPU_TRANSFORMERS_STRUCTURE_CONTEXT_CONTEXT_H_
#define CPU_TRANSFORMERS_STRUCTURE_CONTEXT_CONTEXT_H_

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "structure/context/attr.h"
#include <memory>
#include <string_view>

namespace cpu_transformers {
namespace context {

class Context {
public:
  Context();
  Context(const Context &context) = delete;
  Context(Context &&context) = delete;
  virtual ~Context() = default;
  static std::shared_ptr<Context> Make();
  mlir::MLIRContext &GetMLIRContext();
  mlir::ModuleOp &GetModule();
  FuncAttr &GetFuncAttr();
#ifdef DEBUG
  void DumpModule(std::string_view filename);
#endif
  void SetModule(mlir::ModuleOp module);
  void SetFuncAttr(FuncAttr &&func_attr);
  std::unique_ptr<mlir::ExecutionEngine> MakeExecutionEngine();

private:
  mlir::MLIRContext mlir_context_;
  mlir::ModuleOp module_;
  std::optional<FuncAttr> func_attr_opt_;
};

} // namespace context
} // namespace cpu_transformers

#endif