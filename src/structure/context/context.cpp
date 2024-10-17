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
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "structure/context/attr.h"
#include "worker/builder.h"
#include "worker/lower.h"
#include "worker/planner.h"
#include "worker/runner.h"
#include "llvm/Support/TargetSelect.h"
#include <memory>
#include <string>
#ifdef DEBUG
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#endif

namespace cpu_transformers {
namespace context {

Context::Context() : std::shared_ptr<ContextImpl>(ContextImpl::Make()) {}

Context::Context(const std::shared_ptr<ContextImpl> &context_impl)
    : std::shared_ptr<ContextImpl>(context_impl) {}

Context::Context(std::shared_ptr<ContextImpl> &&context_impl)
    : std::shared_ptr<ContextImpl>(std::move(context_impl)) {}

Context &Context::operator=(const std::shared_ptr<ContextImpl> &context_impl) {
  std::shared_ptr<ContextImpl>::operator=(context_impl);
  return *this;
}

Context &Context::operator=(std::shared_ptr<ContextImpl> &&context_impl) {
  std::shared_ptr<ContextImpl>::operator=(std::move(context_impl));
  return *this;
}

std::unique_ptr<worker::GeneralBuilder>
Context::MakePlaingGeneralBuilder(std::string &&function_name) {
  context::Context context = *this;
  return worker::GeneralBuilder::MakePlain(std::move(function_name),
                                           std::move(context));
}

std::unique_ptr<worker::GeneralBuilder>
Context::MakeDynamicProgrammingGeneralBuilder(std::string &&function_name) {
  context::Context context = *this;
  return worker::GeneralBuilder::MakeDynamicProgramming(
      std::move(function_name), std::move(context));
}

std::unique_ptr<worker::KernelBuilder>
Context::MakeKernelBuilder(std::string &&function_name) {
  context::Context context = *this;
  return worker::KernelBuilder::Make(std::move(function_name),
                                     std::move(context));
}

std::unique_ptr<worker::PlainLinearPlanner> Context::MakePlainLinearPlanner() {
  context::Context context = *this;
  return worker::PlainLinearPlanner::Make(std::move(context));
}

std::unique_ptr<worker::PlainGreedyPlanner> Context::MakePlainGreedyPlanner() {
  context::Context context = *this;
  return worker::PlainGreedyPlanner::Make(std::move(context));
}

std::unique_ptr<worker::DPGreedyPlanner> Context::MakeDPGreedyPlanner() {
  context::Context context = *this;
  return worker::DPGreedyPlanner::Make(std::move(context));
}

std::unique_ptr<worker::Lower> Context::MakeLower() {
  context::Context context = *this;
  return worker::Lower::Make(std::move(context));
}

std::unique_ptr<worker::Runner> Context::MakeRunner() {
  context::Context context = *this;
  return worker::Runner::Make(std::move(context));
}

std::ostream &operator<<(std::ostream &os, Context &context) {
  os << *context;
  return os;
}

std::unique_ptr<ContextImpl> ContextImpl::Make() {
  return std::unique_ptr<ContextImpl>(new ContextImpl);
}

mlir::MLIRContext &ContextImpl::GetMLIRContext() { return mlir_context_; }

mlir::ModuleOp ContextImpl::GetModule() {
  mlir::ModuleOp p = module_.get();
  return p;
}

FuncAttr &ContextImpl::GetFuncAttr() {
#ifdef DEBUG
  assert(func_attr_opt_);
#endif
  return *func_attr_opt_;
}

Factory &ContextImpl::GetFactory() {
  if (!factory_) {
    factory_ = Factory::Make();
  }
  return *factory_;
}

void ContextImpl::SetModule(mlir::OwningOpRef<mlir::ModuleOp> &&module) {
  module_ = std::move(module);
}

void ContextImpl::SetFuncAttr(FuncAttr &&func_attr) {
  func_attr_opt_ = std::move(func_attr);
}

std::unique_ptr<mlir::ExecutionEngine> ContextImpl::MakeExecutionEngine() {
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

std::string ContextImpl::ExportHeaderFile() {
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

std::ostream &operator<<(std::ostream &os, ContextImpl &context_impl) {
  std::string str = "";
  llvm::raw_string_ostream llvm_os(str);
  context_impl.module_->print(llvm_os);
  os << str;
  return os;
}

ContextImpl::ContextImpl() : module_(nullptr), func_attr_opt_(std::nullopt) {
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

} // namespace context
} // namespace cpu_transformers
