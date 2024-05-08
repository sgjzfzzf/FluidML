#include "worker/builder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "structure/context/attr.h"
#include "structure/flow/region.h"
#include "utils/isa.hpp"
#include "utils/type.h"
#include <string>
#ifdef DEBUG
#include "exception/unreachable_exception.h"
#include <cassert>
#endif

namespace cpu_transformers {
namespace worker {

Builder::Builder(std::string &&function_name,
                 std::shared_ptr<context::Context> context)
    : function_name_(std::move(function_name)),
      context_(context ? std::move(context) : context::Context::Make()) {}

void Builder::Run(const flow::Sequence &sequence, const memory::Index &index) {
  mlir::MLIRContext &mlir_context = context_->GetMLIRContext();
  mlir::OpBuilder builder(&mlir_context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
  std::unordered_map<std::string, mlir::Value> symbol_table;
  int64_t buffer_size = index.GetMaximum();
  std::string function_name = function_name_;
  context::FuncAttr func_attr(std::move(function_name), buffer_size);
  const std::vector<std::shared_ptr<flow::Region>> &regions =
      sequence.GetRegions();
  std::vector<std::shared_ptr<flow::InterfaceRegion>> interface_regions;
  std::vector<std::shared_ptr<flow::InnerRegion>> inner_regions;
  size_t interface_regions_num = 0;
  for (std::shared_ptr<flow::Region> region : regions) {
    if (std::shared_ptr<flow::InterfaceRegion> interface_region =
            std::dynamic_pointer_cast<flow::InterfaceRegion>(region)) {
      ++interface_regions_num;
      interface_regions.push_back(std::move(interface_region));
    } else if (std::shared_ptr<flow::InnerRegion> inner_region =
                   std::dynamic_pointer_cast<flow::InnerRegion>(region)) {
      inner_regions.push_back(std::move(inner_region));
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
  }
  // For the inputs, the first element is the buffer, and the followings are
  // the input and output buffers.
  llvm::SmallVector<mlir::Type> input_types;
  for (std::shared_ptr<flow::InterfaceRegion> interface_region :
       interface_regions) {
    std::string name = interface_region->GetName();
    const Meta &meta = interface_region->GetMeta();
    const std::vector<int64_t> &shape = meta.GetShape();
    mlir::Type type = GetMLIRType(meta.GetType(), builder);
    mlir::MemRefType memref_type = mlir::MemRefType::get(shape, type);
    input_types.push_back(memref_type);
    context::ArgumentAttr::Type arg_type;
    if (isa<flow::InputRegion>(interface_region)) {
      arg_type = context::ArgumentAttr::Type::Input;
    } else if (isa<flow::OutputRegion>(interface_region)) {
      arg_type = context::ArgumentAttr::Type::Output;
    } else {
#ifdef DEBUG
      throw UnreachableException();
#else
      __builtin_unreachable();
#endif
    }
    func_attr.PutArgument(context::ArgumentAttr(arg_type, std::move(name),
                                                std::move(memref_type)));
  }
  mlir::MemRefType type =
      mlir::MemRefType::get({buffer_size}, builder.getI8Type());
  input_types.push_back(type);
  mlir::FunctionType function_type =
      mlir::FunctionType::get(&mlir_context, input_types, {});
  mlir::func::FuncOp function = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), function_name_, function_type);
  function->setAttr(mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    builder.getUnitAttr());
  mlir::Block *block = function.addEntryBlock();
#ifdef DEBUG
  assert(block->getNumArguments() == interface_regions_num + 1);
#endif
  // Put the inputs and outputs into the symbol table.
  for (size_t i = 0; i < interface_regions_num; ++i) {
    mlir::BlockArgument arg = block->getArgument(i);
    std::shared_ptr<flow::InterfaceRegion> interface_region =
        interface_regions[i];
    std::string name = interface_region->GetName();
    symbol_table.insert({std::move(name), std::move(arg)});
  }
  mlir::BlockArgument buffer_arg = block->getArgument(interface_regions_num);
  builder.setInsertionPointToStart(block);
  // Allocate memory for the inner regions and put them into the symbol table.
  for (std::shared_ptr<flow::InnerRegion> inner_region : inner_regions) {
    std::string name = inner_region->GetName();
    const Meta &meta = inner_region->GetMeta();
    Type type = meta.GetType();
    const std::vector<int64_t> &shape = meta.GetShape();
    const size_t offset = index.Get(name);
    mlir::arith::ConstantOp offset_constant =
        builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(),
                                                builder.getIndexType(),
                                                builder.getIndexAttr(offset));
    mlir::MemRefType memref_type =
        mlir::MemRefType::get(shape, GetMLIRType(type, builder));
    mlir::Value value = builder.create<mlir::memref::ViewOp>(
        builder.getUnknownLoc(), memref_type, buffer_arg, offset_constant,
        mlir::ValueRange{});
    symbol_table.insert({std::move(name), std::move(value)});
  }
  // Allocate memory for the nodes and put them into the symbol table.
  for (std::shared_ptr<flow::Node> node : sequence.GetNodes()) {
    const int64_t buffer_size = node->GetBufferSize();
    if (buffer_size > 0) {
      std::string name = node->GetName();
      const int64_t offset = index.Get(name);
      mlir::arith::ConstantOp offset_constant =
          builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(),
                                                  builder.getIndexType(),
                                                  builder.getIndexAttr(offset));
      mlir::MemRefType memref_type = mlir::MemRefType::get(
          llvm::ArrayRef<int64_t>{buffer_size}, builder.getI8Type());
      mlir::Value value = builder.create<mlir::memref::ViewOp>(
          builder.getUnknownLoc(), memref_type, buffer_arg, offset_constant,
          mlir::ValueRange{});
      symbol_table.insert({std::move(name), std::move(value)});
    }
  }
  // Run the scheduler.
  getScheduler().Run(builder, sequence, symbol_table);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  module.push_back(function);
#ifdef DEBUG
  assert(mlir::verify(module).succeeded());
#endif
  context_->SetModule(module);
  context_->SetFuncAttr(std::move(func_attr));
}

NaiveBuilder::NaiveBuilder(std::string &&function_name,
                           std::shared_ptr<context::Context> context)
    : Builder(std::move(function_name), std::move(context)),
      scheduler_(NaiveScheduler()) {}

Scheduler &NaiveBuilder::getScheduler() { return scheduler_; }

} // namespace worker
} // namespace cpu_transformers
