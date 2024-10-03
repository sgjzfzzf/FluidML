#include "structure/context/context.h"
#include "fmt/core.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "structure/context/attr.h"
#include "llvm/Support/TargetSelect.h"
#include <memory>
#ifdef DEBUG
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <system_error>
#endif

namespace cpu_transformers {
namespace context {

Context::Context() : module_(nullptr), func_attr_opt_(std::nullopt) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir_context_
      .loadDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                   mlir::bufferization::BufferizationDialect,
                   mlir::BuiltinDialect, mlir::cf::ControlFlowDialect,
                   mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                   mlir::LLVM::LLVMDialect, mlir::math::MathDialect,
                   mlir::memref::MemRefDialect, mlir::omp::OpenMPDialect,
                   mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  mlir::registerBuiltinDialectTranslation(mlir_context_);
  mlir::registerLLVMDialectTranslation(mlir_context_);
  mlir::registerOpenMPDialectTranslation(mlir_context_);
  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  mlir_context_.appendDialectRegistry(registry);
}

std::shared_ptr<Context> Context::Make() { return std::make_shared<Context>(); }

mlir::MLIRContext &Context::GetMLIRContext() { return mlir_context_; }

mlir::ModuleOp Context::GetModule() {
  mlir::ModuleOp p = module_.get();
  return p;
}

FuncAttr &Context::GetFuncAttr() {
#ifdef DEBUG
  assert(func_attr_opt_);
#endif
  return *func_attr_opt_;
}

#ifdef DEBUG
void Context::DumpModule(std::string_view filename) {
  std::error_code ec;
  llvm::raw_fd_ostream file(filename.data(), ec);
  assert(!ec);
  module_->print(file);
}
#endif

void Context::SetModule(mlir::OwningOpRef<mlir::ModuleOp> &&module) {
  module_ = std::move(module);
}

void Context::SetFuncAttr(FuncAttr &&func_attr) {
  func_attr_opt_ = std::move(func_attr);
}

std::unique_ptr<mlir::ExecutionEngine> Context::MakeExecutionEngine() {
  mlir::ExecutionEngineOptions engine_options;
  engine_options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> maybe_engine =
      mlir::ExecutionEngine::create(*module_, engine_options);
#ifdef DEBUG
  assert(maybe_engine);
#endif
  std::unique_ptr<mlir::ExecutionEngine> engine = nullptr;
  maybe_engine->swap(engine);
#ifdef DEBUG
  assert(engine != nullptr);
#endif
  return engine;
}

std::string Context::ExportHeaderFile() {
  if (!func_attr_opt_) {
    return "";
  }
  const FuncAttr &func_attr = *func_attr_opt_;
  const std::string &func_name = func_attr.GetName();
  const std::vector<ArgumentAttr> &argument_attrs = func_attr.GetArguments();
  std::string notes = "";
  std::string params = "";
  for (const ArgumentAttr &argument_attr : argument_attrs) {
    ArgumentAttr::Type type = argument_attr.GetType();
#ifdef DEBUG
    assert(type == ArgumentAttr::Type::Input ||
           type == ArgumentAttr::Type::Output);
#endif
    const std::string &argument_name = argument_attr.GetName();
    const mlir::MemRefType &memref_type = argument_attr.GetMemRefType();
    const mlir::Type &elem_type = memref_type.getElementType();
    llvm::ArrayRef<int64_t> shape = memref_type.getShape();
    std::string shape_str = "";
    std::string param = "";
    for (int64_t dim : shape) {
      if (shape_str.empty()) {
        shape_str = std::to_string(dim);
      } else {
        shape_str += "x" + std::to_string(dim);
      }
      param = "void*, void*, int64_t";
      for (int64_t _ : shape) {
        param += ", int64_t, int64_t";
      }
    }
    std::string elem_type_str = "";
    if (elem_type.isa<mlir::IntegerType>()) {
      elem_type_str = "i" + std::to_string(elem_type.getIntOrFloatBitWidth());
    } else if (elem_type.isa<mlir::FloatType>()) {
      elem_type_str = "f" + std::to_string(elem_type.getIntOrFloatBitWidth());
    } else {
#ifdef DEBUG
      assert(false && "unsupported element type");
#else
      __builtin_unreachable();
#endif
    }
    notes += fmt::format("// {}: {}, {}x{}\n",
                         type == ArgumentAttr::Type::Input ? "input" : "output",
                         argument_name, shape_str, elem_type_str);
    if (params.empty()) {
      params = param;
    } else {
      params += ", " + param;
    }
  }
  const size_t buffer_size = func_attr.GetBuffer();
  notes += fmt::format("// buffer: {}xi8", buffer_size);
  std::string code = fmt::format(R"(
#include <stddef.h>

{}
size_t {}({});
  )",
                                 notes, func_name, params);
  return code;
}

} // namespace context
} // namespace cpu_transformers
