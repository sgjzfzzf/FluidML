#include "worker/builder.h"
#include "fmt/format.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "structure/context/attr.h"
#include "structure/context/factory.h"
#include "structure/flow/node.h"
#include "structure/flow/region.h"
#include "structure/kernel/kernel/kernel.h"
#include "utils/float.h"
#include "utils/isa.hpp"
#include "utils/type.h"
#include "utils/utils.h"
#include "worker/utils.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unistd.h>
#ifdef DEBUG
#include <cassert>
#endif

namespace fluidml {
namespace worker {

class GeneralBuilderImpl : public GeneralBuilder {
public:
  GeneralBuilderImpl(std::string &&function_name, context::Context &&context);
  GeneralBuilderImpl(const GeneralBuilderImpl &builder) = delete;
  GeneralBuilderImpl(GeneralBuilderImpl &&builder) = default;
  virtual ~GeneralBuilderImpl() = default;
  void Run(const flow::Sequence &sequence, const memory::Index &index) override;

protected:
  virtual void
  schedule(mlir::OpBuilder &builder, const flow::Sequence &sequence,
           std::unordered_map<std::string, mlir::Value> &symbol_table) = 0;
  const std::string function_name_;
  context::Context context_;
};

class PlainGeneralBuilderImpl : public GeneralBuilderImpl {
public:
  PlainGeneralBuilderImpl(std::string &&function_name,
                          context::Context &&context);
  PlainGeneralBuilderImpl(const PlainGeneralBuilderImpl &builder) = delete;
  PlainGeneralBuilderImpl(PlainGeneralBuilderImpl &&builder) = default;
  virtual ~PlainGeneralBuilderImpl() = default;

private:
  void
  schedule(mlir::OpBuilder &builder, const flow::Sequence &sequence,
           std::unordered_map<std::string, mlir::Value> &symbol_table) override;
};

class DynamicProgrammingGeneralBuilderImpl : public GeneralBuilderImpl {
public:
  DynamicProgrammingGeneralBuilderImpl(std::string &&function_name,
                                       context::Context &&context);
  DynamicProgrammingGeneralBuilderImpl(
      const DynamicProgrammingGeneralBuilderImpl &builder) = delete;
  DynamicProgrammingGeneralBuilderImpl(
      DynamicProgrammingGeneralBuilderImpl &&builder) = default;
  virtual ~DynamicProgrammingGeneralBuilderImpl() = default;

private:
  void
  schedule(mlir::OpBuilder &builder, const flow::Sequence &sequence,
           std::unordered_map<std::string, mlir::Value> &symbol_table) override;
};

class KernelBuilderImpl : public KernelBuilder {
public:
  KernelBuilderImpl(std::string &&function_name, context::Context &&context);
  KernelBuilderImpl(const KernelBuilderImpl &builder) = delete;
  KernelBuilderImpl(KernelBuilderImpl &&builder) = default;
  virtual ~KernelBuilderImpl() = default;
  void RunOnSingleInputWithoutBuffer(
      const kernel::SingleInputWithoutBufferKernel &kernel,
      const Meta &input_meta, const Meta &output_meta) override;
  void RunOnSingleInputWithoutBuffer(
      const kernel::SingleInputWithoutBufferKernel &kernel,
      const Meta &input_meta, const std::vector<size_t> &input_layout,
      const Meta &output_meta,
      const std::vector<size_t> &output_layout) override;
  void
  RunOnSingleInputWithBuffer(const kernel::SingleInputWithBufferKernel &kernel,
                             const Meta &input_meta, const Meta &output_meta,
                             size_t buffer_size) override;
  void RunOnSingleInputWithBuffer(
      const kernel::SingleInputWithBufferKernel &kernel, const Meta &input_meta,
      const std::vector<size_t> &input_layout, const Meta &output_meta,
      const std::vector<size_t> &output_layout, size_t buffer_size) override;
  void RunOnDoubleInputsWithoutBuffer(
      const kernel::DoubleInputsWithoutBufferKernel &kernel,
      const Meta &lhs_meta, const Meta &rhs_meta,
      const Meta &output_meta) override;
  void RunOnDoubleInputsWithoutBuffer(
      const kernel::DoubleInputsWithoutBufferKernel &kernel,
      const Meta &lhs_meta, const std::vector<size_t> &lhs_layout,
      const Meta &rhs_meta, const std::vector<size_t> &rhs_layout,
      const Meta &output_meta,
      const std::vector<size_t> &output_layout) override;
  void RunOnDoubleInputsWithBuffer(
      const kernel::DoubleInputsWithBufferKernel &kernel, const Meta &lhs_meta,
      const Meta &rhs_meta, const Meta &output_meta,
      size_t buffer_size) override;
  void RunOnDoubleInputsWithBuffer(
      const kernel::DoubleInputsWithBufferKernel &kernel, const Meta &lhs_meta,
      const std::vector<size_t> &lhs_layout, const Meta &rhs_meta,
      const std::vector<size_t> &rhs_layout, const Meta &output_meta,
      const std::vector<size_t> &output_layout, size_t buffer_size) override;

private:
  std::string function_name_;
  context::Context context_;
};

std::unique_ptr<GeneralBuilder>
GeneralBuilder::MakePlain(std::string &&function_name,
                          context::Context &&context) {
  return std::make_unique<PlainGeneralBuilderImpl>(std::move(function_name),
                                                   std::move(context));
}

std::unique_ptr<GeneralBuilder>
GeneralBuilder::MakeDynamicProgramming(std::string &&function_name,
                                       context::Context &&context) {
  return std::make_unique<DynamicProgrammingGeneralBuilderImpl>(
      std::move(function_name), std::move(context));
}

GeneralBuilderImpl::GeneralBuilderImpl(std::string &&function_name,
                                       context::Context &&context)
    : function_name_(std::move(function_name)), context_(context) {}

DynamicProgrammingGeneralBuilderImpl::DynamicProgrammingGeneralBuilderImpl(
    std::string &&function_name, context::Context &&context)
    : GeneralBuilderImpl(std::move(function_name), std::move(context)) {}

void GeneralBuilderImpl::Run(const flow::Sequence &sequence,
                             const memory::Index &index) {
  mlir::MLIRContext &mlir_context = context_->GetMLIRContext();
  mlir::OpBuilder builder(&mlir_context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  std::unordered_map<std::string, mlir::Value> symbol_table;
  int64_t buffer_size = index.GetMaximum();
  std::string function_name = function_name_;
  context::FuncAttr func_attr(std::move(function_name), buffer_size);
  const std::vector<std::shared_ptr<flow::Region>> &regions =
      sequence.GetRegions();
  std::vector<std::shared_ptr<flow::InterfaceRegion>> interface_regions;
  std::vector<std::shared_ptr<flow::InnerRegion>> inner_regions;
  std::vector<std::shared_ptr<flow::ConstantRegion>> constant_regions;
  for (std::shared_ptr<flow::Region> region : regions) {
    if (std::shared_ptr<flow::InterfaceRegion> interface_region =
            std::dynamic_pointer_cast<flow::InterfaceRegion>(region)) {
      interface_regions.push_back(std::move(interface_region));
    } else if (std::shared_ptr<flow::InnerRegion> inner_region =
                   std::dynamic_pointer_cast<flow::InnerRegion>(region)) {
      inner_regions.push_back(std::move(inner_region));
    } else if (std::shared_ptr<flow::ConstantRegion> constant_region =
                   std::dynamic_pointer_cast<flow::ConstantRegion>(region)) {
      constant_regions.push_back(std::move(constant_region));
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
  }
  const size_t interface_regions_num = interface_regions.size();
  mlir::SymbolTable sym_table(*module);
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
      assert(false && "unreachable");
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
    const std::vector<size_t> &layout = inner_region->GetLayout();
    const size_t offset = index.Get(name);
    mlir::arith::ConstantOp offset_constant =
        builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(),
                                                builder.getIndexType(),
                                                builder.getIndexAttr(offset));
    const size_t shape_len = shape.size();
#ifdef DEBUG
    assert(shape_len == layout.size());
#endif
    llvm::SmallVector<mlir::AffineExpr> layout_exprs;
    for (size_t dim : layout) {
      layout_exprs.push_back(builder.getAffineDimExpr(dim));
    }
    const std::vector<int64_t> &physical_shape =
                                   inner_region->GetPhysicalShape(),
                               &strides = inner_region->GetStrides();
    mlir::StridedLayoutAttr strided_layout =
        mlir::StridedLayoutAttr::get(&mlir_context, 0, strides);
    mlir::MemRefType plain_memref_type = mlir::MemRefType::get(
                         physical_shape, GetMLIRType(type, builder)),
                     strided_memref_type = mlir::MemRefType::get(
                         shape, GetMLIRType(type, builder), strided_layout);
    mlir::Value view_op = builder.create<mlir::memref::ViewOp>(
                    builder.getUnknownLoc(), plain_memref_type, buffer_arg,
                    offset_constant, mlir::ValueRange{}),
                reinterpret_cast_op =
                    builder.create<mlir::memref::ReinterpretCastOp>(
                        builder.getUnknownLoc(), strided_memref_type, view_op,
                        0, shape, strides);
    symbol_table.insert({std::move(name), std::move(reinterpret_cast_op)});
  }
  // Create `memref.global` for the constant regions and put the corresponding
  // `memref.get_global` into the symbol table.
  for (std::shared_ptr<flow::ConstantRegion> constant_region :
       constant_regions) {
    std::string name = constant_region->GetName();
    const Tensor &tensor = constant_region->GetTensor();
    const std::vector<int64_t> &shape = tensor.GetShape();
    const std::vector<float64_t> &buffer = tensor.Get();
    const std::vector<size_t> &layout = constant_region->GetLayout();
    std::vector<int64_t> strides = utils::GenStrides(shape, layout);
    Type type = tensor.GetType();
    mlir::OpBuilder::InsertPoint ip = builder.saveInsertionPoint();
    builder.setInsertionPoint(*module);
    mlir::MemRefType plain_memref_type;
    if (type == Type::kFloat32) {
      plain_memref_type = mlir::MemRefType::get(shape, builder.getF32Type());
      mlir::RankedTensorType tensor_type =
          mlir::RankedTensorType::get(shape, builder.getF32Type());
      llvm::SmallVector<float32_t> data;
      for (const std::vector<size_t> &indices :
           utils::GenAllIndicesInOrder(shape)) {
        size_t index = utils::GenIndex(indices, strides);
#ifdef DEBUG
        assert(index < buffer.size());
#endif
        data.push_back(buffer[index]);
      }
      mlir::DenseElementsAttr elements =
          mlir::DenseElementsAttr::get(tensor_type, llvm::ArrayRef(data));
      mlir::memref::GlobalOp global_op = builder.create<mlir::memref::GlobalOp>(
          builder.getUnknownLoc(), name, builder.getStringAttr("private"),
          plain_memref_type, elements, true, mlir::IntegerAttr());
      sym_table.insert(global_op);
    } else if (type == Type::kInt64) {
      plain_memref_type = mlir::MemRefType::get(shape, builder.getI64Type());
      mlir::RankedTensorType tensor_type =
          mlir::RankedTensorType::get(shape, builder.getI64Type());
      llvm::SmallVector<int64_t> data;
      for (const std::vector<size_t> &indices :
           utils::GenAllIndicesInOrder(shape)) {
        size_t index = utils::GenIndex(indices, strides);
#ifdef DEBUG
        assert(index < buffer.size());
#endif
        data.push_back(buffer[index]);
      }
      mlir::DenseElementsAttr elements =
          mlir::DenseElementsAttr::get(tensor_type, llvm::ArrayRef(data));
      mlir::memref::GlobalOp global_op = builder.create<mlir::memref::GlobalOp>(
          builder.getUnknownLoc(), name, builder.getStringAttr("private"),
          plain_memref_type, elements, true, mlir::IntegerAttr());
      sym_table.insert(global_op);
    } else if (type == Type::kBool) {
      plain_memref_type = mlir::MemRefType::get(shape, builder.getI1Type());
      mlir::RankedTensorType tensor_type =
          mlir::RankedTensorType::get(shape, builder.getI1Type());
      llvm::SmallVector<bool> data;
      for (const std::vector<size_t> &indices :
           utils::GenAllIndicesInOrder(shape)) {
        size_t index = utils::GenIndex(indices, strides);
#ifdef DEBUG
        assert(index < buffer.size());
#endif
        data.push_back(buffer[index]);
      }
      mlir::DenseElementsAttr elements =
          mlir::DenseElementsAttr::get(tensor_type, llvm::ArrayRef(data));
      mlir::memref::GlobalOp global_op = builder.create<mlir::memref::GlobalOp>(
          builder.getUnknownLoc(), name, builder.getStringAttr("private"),
          plain_memref_type, elements, true, mlir::IntegerAttr());
      sym_table.insert(global_op);
    } else {
#ifdef DEBUG
      assert(false && "unimplemented");
#else
      __builtin_unreachable();
#endif
    }
    builder.restoreInsertionPoint(ip);
    mlir::StridedLayoutAttr strided_layout =
        mlir::StridedLayoutAttr::get(&mlir_context, 0, strides);
    mlir::MemRefType memref_type = mlir::MemRefType::get(
        shape, plain_memref_type.getElementType(), strided_layout);
    mlir::Value get_op = builder.create<mlir::memref::GetGlobalOp>(
                    builder.getUnknownLoc(), plain_memref_type, name),
                reinterpret_cast_op =
                    builder.create<mlir::memref::ReinterpretCastOp>(
                        builder.getUnknownLoc(), memref_type, get_op, 0, shape,
                        strides);
    symbol_table.insert({std::move(name), std::move(reinterpret_cast_op)});
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
      symbol_table.insert({GetBufferName(name), std::move(value)});
    }
  }
  // Run the schedule.
  schedule(builder, sequence, symbol_table);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  module->push_back(function);
#ifdef DEBUG
  assert(
#endif
      mlir::verify(*module).succeeded()
#ifdef DEBUG
  )
#endif
      ;
  context_->SetModule(std::move(module));
  context_->SetFuncAttr(std::move(func_attr));
}

PlainGeneralBuilderImpl::PlainGeneralBuilderImpl(std::string &&function_name,
                                                 context::Context &&context)
    : GeneralBuilderImpl(std::move(function_name), std::move(context)) {}

void PlainGeneralBuilderImpl::schedule(
    mlir::OpBuilder &builder, const flow::Sequence &sequence,
    std::unordered_map<std::string, mlir::Value> &symbol_table) {
  const std::vector<std::shared_ptr<flow::Node>> &nodes = sequence.GetNodes();
  for (std::shared_ptr<flow::Node> node : nodes) {
    std::shared_ptr<kernel::Kernel> mkernel = worker::SelectKernel(node.get());
#ifdef DEBUG
    assert(mkernel != nullptr);
#endif
    if (std::shared_ptr<flow::SingleInputWithoutBufferNode> ptr =
            std::dynamic_pointer_cast<flow::SingleInputWithoutBufferNode>(
                node)) {
      std::shared_ptr<kernel::SingleInputWithoutBufferKernel> kernel =
          std::dynamic_pointer_cast<kernel::SingleInputWithoutBufferKernel>(
              mkernel);
#ifdef DEBUG
      assert(kernel != nullptr);
#endif
      const std::string &input_name = ptr->GetInputAsString(),
                        &output_name = ptr->GetOutputAsString();
      mlir::Value &input = symbol_table.at(input_name),
                  &output = symbol_table.at(output_name);
      std::shared_ptr<flow::Region> input_region = ptr->GetInput(),
                                    output_region = ptr->GetOutput();
      kernel->Run(builder, input, output);
    } else if (std::shared_ptr<flow::SingleInputWithBufferNode> ptr =
                   std::dynamic_pointer_cast<flow::SingleInputWithBufferNode>(
                       node)) {
      std::shared_ptr<kernel::SingleInputWithBufferKernel> kernel =
          std::dynamic_pointer_cast<kernel::SingleInputWithBufferKernel>(
              mkernel);
#ifdef DEBUG
      assert(kernel != nullptr);
#endif
      const std::string &input_name = ptr->GetInputAsString(),
                        &output_name = ptr->GetOutputAsString();
      std::string buffer_name = GetBufferName(ptr->GetName());
      mlir::Value &input = symbol_table.at(input_name),
                  &output = symbol_table.at(output_name),
                  &buffer = symbol_table.at(buffer_name);
      std::shared_ptr<flow::Region> input_region = ptr->GetInput(),
                                    output_region = ptr->GetOutput();
      kernel->Run(builder, input, output, buffer);
    } else if (std::shared_ptr<flow::DoubleInputsWithoutBufferNode> ptr =
                   std::dynamic_pointer_cast<
                       flow::DoubleInputsWithoutBufferNode>(node)) {
      std::shared_ptr<kernel::DoubleInputsWithoutBufferKernel> kernel =
          std::dynamic_pointer_cast<kernel::DoubleInputsWithoutBufferKernel>(
              mkernel);
#ifdef DEBUG
      assert(kernel != nullptr);
#endif
      const std::string &lhs_name = ptr->GetLhsAsString(),
                        &rhs_name = ptr->GetRhsAsString(),
                        &output_name = ptr->GetOutputAsString();
      mlir::Value &lhs = symbol_table.at(lhs_name),
                  &rhs = symbol_table.at(rhs_name),
                  &output = symbol_table.at(output_name);
      std::shared_ptr<flow::Region> lhs_region = ptr->GetLhs(),
                                    rhs_region = ptr->GetRhs(),
                                    output_region = ptr->GetOutput();
      kernel->Run(builder, lhs, rhs, output);
    } else if (std::shared_ptr<flow::DoubleInputsWithBufferNode> ptr =
                   std::dynamic_pointer_cast<flow::DoubleInputsWithBufferNode>(
                       node)) {
      std::shared_ptr<kernel::DoubleInputsWithBufferKernel> kernel =
          std::dynamic_pointer_cast<kernel::DoubleInputsWithBufferKernel>(
              mkernel);
#ifdef DEBUG
      assert(kernel != nullptr);
#endif
      const std::string &lhs_name = ptr->GetLhsAsString(),
                        &rhs_name = ptr->GetRhsAsString(),
                        &output_name = ptr->GetOutputAsString();
      std::string buffer_name = GetBufferName(ptr->GetName());
      mlir::Value &lhs = symbol_table.at(lhs_name),
                  &rhs = symbol_table.at(rhs_name),
                  &output = symbol_table.at(output_name),
                  &buffer = symbol_table.at(buffer_name);
      std::shared_ptr<flow::Region> lhs_region = ptr->GetLhs(),
                                    rhs_region = ptr->GetRhs(),
                                    output_region = ptr->GetOutput();
      kernel->Run(builder, lhs, rhs, output, buffer);
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
  }
}

void DynamicProgrammingGeneralBuilderImpl::schedule(
    mlir::OpBuilder &builder, const flow::Sequence &sequence,
    std::unordered_map<std::string, mlir::Value> &symbol_table) {
  const std::vector<std::shared_ptr<flow::Node>> &nodes = sequence.GetNodes();
  context::Factory &factory = context_->GetFactory();
  for (std::shared_ptr<flow::Node> node : nodes) {
    std::shared_ptr<kernel::KernelGenerator> kgenerator =
        factory.MakeKernelGenerator(*node);
#ifdef DEBUG
    assert(kgenerator != nullptr);
#endif
    if (std::shared_ptr<flow::SingleInputWithoutBufferNode> ptr =
            std::dynamic_pointer_cast<flow::SingleInputWithoutBufferNode>(
                node)) {
      std::shared_ptr<kernel::SingleInputWithoutBufferKernelGenerator>
          generator = std::dynamic_pointer_cast<
              kernel::SingleInputWithoutBufferKernelGenerator>(kgenerator);
#ifdef DEBUG
      assert(generator != nullptr);
#endif
      const std::string &input_name = ptr->GetInputAsString(),
                        &output_name = ptr->GetOutputAsString();
      mlir::Value &input = symbol_table.at(input_name),
                  &output = symbol_table.at(output_name);
      std::shared_ptr<flow::Region> input_region = ptr->GetInput(),
                                    output_region = ptr->GetOutput();
      llvm::ArrayRef<size_t> input_layout = input_region->GetLayout(),
                             output_layout = output_region->GetLayout();
      std::shared_ptr<kernel::SingleInputWithoutBufferKernel> kernel =
          generator->YieldSingleInputWithoutBufferKernel(input_layout,
                                                         output_layout);
      kernel->Run(builder, input, output);
    } else if (std::shared_ptr<flow::SingleInputWithBufferNode> ptr =
                   std::dynamic_pointer_cast<flow::SingleInputWithBufferNode>(
                       node)) {
      std::shared_ptr<kernel::SingleInputWithBufferKernelGenerator> generator =
          std::dynamic_pointer_cast<
              kernel::SingleInputWithBufferKernelGenerator>(kgenerator);
#ifdef DEBUG
      assert(generator != nullptr);
#endif
      const std::string &input_name = ptr->GetInputAsString(),
                        &output_name = ptr->GetOutputAsString();
      std::string buffer_name = GetBufferName(ptr->GetName());
      mlir::Value &input = symbol_table.at(input_name),
                  &output = symbol_table.at(output_name),
                  &buffer = symbol_table.at(buffer_name);
      std::shared_ptr<flow::Region> input_region = ptr->GetInput(),
                                    output_region = ptr->GetOutput();
      llvm::ArrayRef<size_t> input_layout = input_region->GetLayout(),
                             output_layout = output_region->GetLayout();
      std::shared_ptr<kernel::SingleInputWithBufferKernel> kernel =
          generator->YieldSingleInputWithBufferKernel(input_layout,
                                                      output_layout);
      kernel->Run(builder, input, output, buffer);
    } else if (std::shared_ptr<flow::DoubleInputsWithoutBufferNode> ptr =
                   std::dynamic_pointer_cast<
                       flow::DoubleInputsWithoutBufferNode>(node)) {
      std::shared_ptr<kernel::DoubleInputsWithoutBufferKernelGenerator>
          generator = std::dynamic_pointer_cast<
              kernel::DoubleInputsWithoutBufferKernelGenerator>(kgenerator);
#ifdef DEBUG
      assert(generator != nullptr);
#endif
      const std::string &lhs_name = ptr->GetLhsAsString(),
                        &rhs_name = ptr->GetRhsAsString(),
                        &output_name = ptr->GetOutputAsString();
      mlir::Value &lhs = symbol_table.at(lhs_name),
                  &rhs = symbol_table.at(rhs_name),
                  &output = symbol_table.at(output_name);
      std::shared_ptr<flow::Region> lhs_region = ptr->GetLhs(),
                                    rhs_region = ptr->GetRhs(),
                                    output_region = ptr->GetOutput();
      llvm::ArrayRef<size_t> lhs_layout = lhs_region->GetLayout(),
                             rhs_layout = rhs_region->GetLayout(),
                             output_layout = output_region->GetLayout();
      std::shared_ptr<kernel::DoubleInputsWithoutBufferKernel> kernel =
          generator->YieldDoubleInputsWithoutBufferKernel(
              lhs_layout, rhs_layout, output_layout);
      kernel->Run(builder, lhs, rhs, output);
    } else if (const std::shared_ptr<flow::DoubleInputsWithBufferNode> ptr =
                   std::dynamic_pointer_cast<flow::DoubleInputsWithBufferNode>(
                       node)) {
      std::shared_ptr<kernel::DoubleInputsWithBufferKernelGenerator> generator =
          std::dynamic_pointer_cast<
              kernel::DoubleInputsWithBufferKernelGenerator>(kgenerator);
#ifdef DEBUG
      assert(generator != nullptr);
#endif
      const std::string &lhs_name = ptr->GetLhsAsString(),
                        &rhs_name = ptr->GetRhsAsString(),
                        &output_name = ptr->GetOutputAsString();
      std::string buffer_name = GetBufferName(ptr->GetName());
      mlir::Value &lhs = symbol_table.at(lhs_name),
                  &rhs = symbol_table.at(rhs_name),
                  &output = symbol_table.at(output_name),
                  &buffer = symbol_table.at(buffer_name);
      std::shared_ptr<flow::Region> lhs_region = ptr->GetLhs(),
                                    rhs_region = ptr->GetRhs(),
                                    output_region = ptr->GetOutput();
      llvm::ArrayRef<size_t> lhs_layout = lhs_region->GetLayout(),
                             rhs_layout = rhs_region->GetLayout(),
                             output_layout = output_region->GetLayout();
      std::shared_ptr<kernel::DoubleInputsWithBufferKernel> kernel =
          generator->YieldDoubleInputsWithBufferKernel(lhs_layout, rhs_layout,
                                                       output_layout);
      kernel->Run(builder, lhs, rhs, output, buffer);
    } else {
#ifdef DEBUG
      assert(false && "unreachable");
#else
      __builtin_unreachable();
#endif
    }
  }
}

std::unique_ptr<KernelBuilder> KernelBuilder::Make(std::string &&function_name,
                                                   context::Context &&context) {
  return std::make_unique<KernelBuilderImpl>(std::move(function_name),
                                             std::move(context));
}

KernelBuilderImpl::KernelBuilderImpl(std::string &&function_name,
                                     context::Context &&context)
    : function_name_(std::move(function_name)), context_(std::move(context)) {}

void KernelBuilderImpl::RunOnSingleInputWithoutBuffer(
    const kernel::SingleInputWithoutBufferKernel &kernel,
    const Meta &input_meta, const Meta &output_meta) {
  const std::vector<int64_t> input_shape = input_meta.GetShape(),
                             output_shape = output_meta.GetShape();
  const size_t input_len = input_shape.size(), output_len = output_shape.size();
  std::vector<size_t> input_layout(input_len), output_layout(output_len);
  for (size_t i = 0; i < input_len; ++i) {
    input_layout[i] = i;
  }
  for (size_t i = 0; i < output_len; ++i) {
    output_layout[i] = i;
  }
  RunOnSingleInputWithoutBuffer(kernel, input_meta, input_layout, output_meta,
                                output_layout);
}

void KernelBuilderImpl::RunOnSingleInputWithoutBuffer(
    const kernel::SingleInputWithoutBufferKernel &kernel,
    const Meta &input_meta, const std::vector<size_t> &input_layout,
    const Meta &output_meta, const std::vector<size_t> &output_layout) {
  mlir::MLIRContext &mlir_context = context_->GetMLIRContext();
  mlir::OpBuilder builder(&mlir_context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  mlir::Type input_elem_type = GetMLIRType(input_meta.GetType(), builder),
             output_elem_type = GetMLIRType(output_meta.GetType(), builder);
  const std::vector<int64_t> &input_shape = input_meta.GetShape(),
                             &output_shape = output_meta.GetShape();
  std::vector<int64_t> input_strides =
                           utils::GenStrides(input_shape, input_layout),
                       output_strides =
                           utils::GenStrides(output_shape, output_layout);
  mlir::StridedLayoutAttr input_strided_layout = mlir::StridedLayoutAttr::get(
                              &mlir_context, 0, input_strides),
                          output_strided_layout = mlir::StridedLayoutAttr::get(
                              &mlir_context, 0, output_strides);
  mlir::MemRefType input_memref_type = mlir::MemRefType::get(
                       input_shape, input_elem_type, input_strided_layout),
                   output_memref_type = mlir::MemRefType::get(
                       output_shape, output_elem_type, output_strided_layout);
  mlir::FunctionType function_type = mlir::FunctionType::get(
      &mlir_context, {input_memref_type, output_memref_type}, {});
  mlir::func::FuncOp function = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), function_name_, function_type);
  function->setAttr(mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    builder.getUnitAttr());
  mlir::Block *block = function.addEntryBlock();
  builder.setInsertionPointToStart(block);
  mlir::Value input = block->getArgument(0);
  mlir::Value output = block->getArgument(1);
  kernel.Run(builder, input, output);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  module->push_back(std::move(function));
#ifdef DEBUG
  assert(
#endif
      mlir::verify(*module).succeeded()
#ifdef DEBUG
  )
#endif
      ;
  context_->SetModule(std::move(module));
  std::string function_name = function_name_;
  context::FuncAttr func_attr(std::move(function_name), 0);
  context::ArgumentAttr input_arg_attr(context::ArgumentAttr::Type::Input,
                                       kInputKey, std::move(input_memref_type)),
      output_arg_attr(context::ArgumentAttr::Type::Output, kOutputKey,
                      std::move(output_memref_type));
  func_attr.PutArgument(std::move(input_arg_attr));
  func_attr.PutArgument(std::move(output_arg_attr));
  context_->SetFuncAttr(std::move(func_attr));
}

void KernelBuilderImpl::RunOnSingleInputWithBuffer(
    const kernel::SingleInputWithBufferKernel &kernel, const Meta &input_meta,
    const Meta &output_meta, size_t buffer_size) {
  const std::vector<int64_t> input_shape = input_meta.GetShape(),
                             output_shape = output_meta.GetShape();
  const size_t input_len = input_shape.size(), output_len = output_shape.size();
  std::vector<size_t> input_layout(input_len), output_layout(output_len);
  for (size_t i = 0; i < input_len; ++i) {
    input_layout[i] = i;
  }
  for (size_t i = 0; i < output_len; ++i) {
    output_layout[i] = i;
  }
  RunOnSingleInputWithBuffer(kernel, input_meta, input_layout, output_meta,
                             output_layout, buffer_size);
}

void KernelBuilderImpl::RunOnSingleInputWithBuffer(
    const kernel::SingleInputWithBufferKernel &kernel, const Meta &input_meta,
    const std::vector<size_t> &input_layout, const Meta &output_meta,
    const std::vector<size_t> &output_layout, size_t buffer_size) {
  mlir::MLIRContext &mlir_context = context_->GetMLIRContext();
  mlir::OpBuilder builder(&mlir_context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  mlir::Type input_elem_type = GetMLIRType(input_meta.GetType(), builder),
             output_elem_type = GetMLIRType(output_meta.GetType(), builder);
  const std::vector<int64_t> &input_shape = input_meta.GetShape(),
                             &output_shape = output_meta.GetShape();
  std::vector<int64_t> input_strides =
                           utils::GenStrides(input_shape, input_layout),
                       output_strides =
                           utils::GenStrides(output_shape, output_layout);
  mlir::StridedLayoutAttr input_strided_layout = mlir::StridedLayoutAttr::get(
                              &mlir_context, 0, input_strides),
                          output_strided_layout = mlir::StridedLayoutAttr::get(
                              &mlir_context, 0, output_strides);
  mlir::MemRefType input_memref_type = mlir::MemRefType::get(
                       input_meta.GetShape(), input_elem_type,
                       input_strided_layout),
                   output_memref_type = mlir::MemRefType::get(
                       output_meta.GetShape(), output_elem_type,
                       output_strided_layout),
                   buffer_memref_type = mlir::MemRefType::get(
                       llvm::ArrayRef<int64_t>{
                           static_cast<int64_t>(buffer_size)},
                       builder.getI8Type());
  mlir::FunctionType function_type = mlir::FunctionType::get(
      &mlir_context,
      {input_memref_type, output_memref_type, buffer_memref_type}, {});
  mlir::func::FuncOp function = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), function_name_, function_type);
  function->setAttr(mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    builder.getUnitAttr());
  mlir::Block *block = function.addEntryBlock();
  builder.setInsertionPointToStart(block);
  mlir::Value input = block->getArgument(0), output = block->getArgument(1),
              buffer = block->getArgument(2);
  kernel.Run(builder, input, output, buffer);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  module->push_back(std::move(function));
#ifdef DEBUG
  assert(mlir::verify(*module).succeeded());
#endif
  context_->SetModule(std::move(module));
  const size_t input_size = input_memref_type.getNumElements(),
               output_size = output_memref_type.getNumElements();
  std::string function_name = function_name_;
  context::FuncAttr func_attr(std::move(function_name), buffer_size);
  context::ArgumentAttr input_arg_attr(context::ArgumentAttr::Type::Input,
                                       kInputKey, std::move(input_memref_type)),
      output_arg_attr(context::ArgumentAttr::Type::Output, kOutputKey,
                      std::move(output_memref_type));
  func_attr.PutArgument(std::move(input_arg_attr));
  func_attr.PutArgument(std::move(output_arg_attr));
  context_->SetFuncAttr(std::move(func_attr));
}

void KernelBuilderImpl::RunOnDoubleInputsWithoutBuffer(
    const kernel::DoubleInputsWithoutBufferKernel &kernel, const Meta &lhs_meta,
    const Meta &rhs_meta, const Meta &output_meta) {
  const std::vector<int64_t> lhs_shape = lhs_meta.GetShape(),
                             rhs_shape = rhs_meta.GetShape(),
                             output_shape = output_meta.GetShape();
  const size_t lhs_len = lhs_shape.size(), rhs_len = rhs_shape.size(),
               output_len = output_shape.size();
  std::vector<size_t> lhs_layout(lhs_len), rhs_layout(rhs_len),
      output_layout(output_len);
  for (size_t i = 0; i < lhs_len; ++i) {
    lhs_layout[i] = i;
  }
  for (size_t i = 0; i < rhs_len; ++i) {
    rhs_layout[i] = i;
  }
  for (size_t i = 0; i < output_len; ++i) {
    output_layout[i] = i;
  }
  RunOnDoubleInputsWithoutBuffer(kernel, lhs_meta, lhs_layout, rhs_meta,
                                 rhs_layout, output_meta, output_layout);
}

void KernelBuilderImpl::RunOnDoubleInputsWithoutBuffer(
    const kernel::DoubleInputsWithoutBufferKernel &kernel, const Meta &lhs_meta,
    const std::vector<size_t> &lhs_layout, const Meta &rhs_meta,
    const std::vector<size_t> &rhs_layout, const Meta &output_meta,
    const std::vector<size_t> &output_layout) {
  mlir::MLIRContext &mlir_context = context_->GetMLIRContext();
  mlir::OpBuilder builder(&mlir_context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  mlir::Type lhs_elem_type = GetMLIRType(lhs_meta.GetType(), builder),
             rhs_elem_type = GetMLIRType(rhs_meta.GetType(), builder),
             output_elem_type = GetMLIRType(output_meta.GetType(), builder);
  const std::vector<int64_t> &lhs_shape = lhs_meta.GetShape(),
                             &rhs_shape = rhs_meta.GetShape(),
                             &output_shape = output_meta.GetShape();
  std::vector<int64_t> lhs_strides = utils::GenStrides(lhs_shape, lhs_layout),
                       rhs_strides = utils::GenStrides(rhs_shape, rhs_layout),
                       output_strides =
                           utils::GenStrides(output_shape, output_layout);
  mlir::StridedLayoutAttr lhs_strided_layout = mlir::StridedLayoutAttr::get(
                              &mlir_context, 0, lhs_strides),
                          rhs_strided_layout = mlir::StridedLayoutAttr::get(
                              &mlir_context, 0, rhs_strides),
                          output_strided_layout = mlir::StridedLayoutAttr::get(
                              &mlir_context, 0, output_strides);
  mlir::MemRefType lhs_memref_type = mlir::MemRefType::get(
                       lhs_meta.GetShape(), lhs_elem_type, lhs_strided_layout),
                   rhs_memref_type = mlir::MemRefType::get(
                       rhs_meta.GetShape(), rhs_elem_type, rhs_strided_layout),
                   output_memref_type = mlir::MemRefType::get(
                       output_meta.GetShape(), output_elem_type,
                       output_strided_layout);
  mlir::FunctionType function_type = mlir::FunctionType::get(
      &mlir_context, {lhs_memref_type, rhs_memref_type, output_memref_type},
      {});
  mlir::func::FuncOp function = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), function_name_, function_type);
  function->setAttr(mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    builder.getUnitAttr());
  mlir::Block *block = function.addEntryBlock();
  builder.setInsertionPointToStart(block);
  mlir::Value lhs = block->getArgument(0);
  mlir::Value rhs = block->getArgument(1);
  mlir::Value output = block->getArgument(2);
  kernel.Run(builder, lhs, rhs, output);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  module->push_back(std::move(function));
#ifdef DEBUG
  assert(mlir::verify(*module).succeeded());
#endif
  context_->SetModule(std::move(module));
  const size_t lhs_size = lhs_memref_type.getNumElements(),
               rhs_size = rhs_memref_type.getNumElements(),
               output_size = output_memref_type.getNumElements();
  std::string function_name = function_name_;
  context::FuncAttr func_attr(std::move(function_name), 0);
  context::ArgumentAttr lhs_arg_attr(context::ArgumentAttr::Type::Input,
                                     kLhsKey, std::move(lhs_memref_type)),
      rhs_arg_attr(context::ArgumentAttr::Type::Input, kRhsKey,
                   std::move(rhs_memref_type)),
      output_arg_attr(context::ArgumentAttr::Type::Output, kOutputKey,
                      std::move(output_memref_type));
  func_attr.PutArgument(std::move(lhs_arg_attr));
  func_attr.PutArgument(std::move(rhs_arg_attr));
  func_attr.PutArgument(std::move(output_arg_attr));
  context_->SetFuncAttr(std::move(func_attr));
}

void KernelBuilderImpl::RunOnDoubleInputsWithBuffer(
    const kernel::DoubleInputsWithBufferKernel &kernel, const Meta &lhs_meta,
    const Meta &rhs_meta, const Meta &output_meta, size_t buffer_size) {
  const std::vector<int64_t> lhs_shape = lhs_meta.GetShape(),
                             rhs_shape = rhs_meta.GetShape(),
                             output_shape = output_meta.GetShape();
  const size_t lhs_len = lhs_shape.size(), rhs_len = rhs_shape.size(),
               output_len = output_shape.size();
  std::vector<size_t> lhs_layout(lhs_len), rhs_layout(rhs_len),
      output_layout(output_len);
  for (size_t i = 0; i < lhs_len; ++i) {
    lhs_layout[i] = i;
  }
  for (size_t i = 0; i < rhs_len; ++i) {
    rhs_layout[i] = i;
  }
  for (size_t i = 0; i < output_len; ++i) {
    output_layout[i] = i;
  }
  RunOnDoubleInputsWithBuffer(kernel, lhs_meta, lhs_layout, rhs_meta,
                              rhs_layout, output_meta, output_layout,
                              buffer_size);
}

void KernelBuilderImpl::RunOnDoubleInputsWithBuffer(
    const kernel::DoubleInputsWithBufferKernel &kernel, const Meta &lhs_meta,
    const std::vector<size_t> &lhs_layout, const Meta &rhs_meta,
    const std::vector<size_t> &rhs_layout, const Meta &output_meta,
    const std::vector<size_t> &output_layout, size_t buffer_size) {
  mlir::MLIRContext &mlir_context = context_->GetMLIRContext();
  mlir::OpBuilder builder(&mlir_context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  mlir::Type lhs_elem_type = GetMLIRType(lhs_meta.GetType(), builder),
             rhs_elem_type = GetMLIRType(rhs_meta.GetType(), builder),
             output_elem_type = GetMLIRType(output_meta.GetType(), builder);
  const std::vector<int64_t> &lhs_shape = lhs_meta.GetShape(),
                             &rhs_shape = rhs_meta.GetShape(),
                             &output_shape = output_meta.GetShape();
  std::vector<int64_t> lhs_strides = utils::GenStrides(lhs_shape, lhs_layout),
                       rhs_strides = utils::GenStrides(rhs_shape, rhs_layout),
                       output_strides =
                           utils::GenStrides(output_shape, output_layout);
  mlir::StridedLayoutAttr lhs_strided_layout = mlir::StridedLayoutAttr::get(
                              &mlir_context, 0, lhs_strides),
                          rhs_strided_layout = mlir::StridedLayoutAttr::get(
                              &mlir_context, 0, rhs_strides),
                          output_strided_layout = mlir::StridedLayoutAttr::get(
                              &mlir_context, 0, output_strides);
  mlir::MemRefType lhs_memref_type = mlir::MemRefType::get(
                       lhs_meta.GetShape(), lhs_elem_type, lhs_strided_layout),
                   rhs_memref_type = mlir::MemRefType::get(
                       rhs_meta.GetShape(), rhs_elem_type, rhs_strided_layout),
                   output_memref_type = mlir::MemRefType::get(
                       output_meta.GetShape(), output_elem_type,
                       output_strided_layout),
                   buffer_memref_type = mlir::MemRefType::get(
                       llvm::ArrayRef<int64_t>{
                           static_cast<int64_t>(buffer_size)},
                       builder.getI8Type());
  mlir::FunctionType function_type =
      mlir::FunctionType::get(&mlir_context,
                              {lhs_memref_type, rhs_memref_type,
                               output_memref_type, buffer_memref_type},
                              {});
  mlir::func::FuncOp function = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), function_name_, function_type);
  function->setAttr(mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    builder.getUnitAttr());
  mlir::Block *block = function.addEntryBlock();
  builder.setInsertionPointToStart(block);
  mlir::Value lhs = block->getArgument(0);
  mlir::Value rhs = block->getArgument(1);
  mlir::Value output = block->getArgument(2);
  mlir::Value buffer = block->getArgument(3);
  kernel.Run(builder, lhs, rhs, output, buffer);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  module->push_back(std::move(function));
#ifdef DEBUG
  assert(mlir::verify(*module).succeeded());
#endif
  context_->SetModule(std::move(module));
  const size_t lhs_size = lhs_memref_type.getNumElements(),
               rhs_size = rhs_memref_type.getNumElements(),
               output_size = output_memref_type.getNumElements();
  std::string function_name = function_name_;
  context::FuncAttr func_attr(std::move(function_name), buffer_size);
  context::ArgumentAttr lhs_arg_attr(context::ArgumentAttr::Type::Input,
                                     kLhsKey, std::move(lhs_memref_type)),
      rhs_arg_attr(context::ArgumentAttr::Type::Input, kRhsKey,
                   std::move(rhs_memref_type)),
      output_arg_attr(context::ArgumentAttr::Type::Output, kOutputKey,
                      std::move(output_memref_type));
  func_attr.PutArgument(std::move(lhs_arg_attr));
  func_attr.PutArgument(std::move(rhs_arg_attr));
  func_attr.PutArgument(std::move(output_arg_attr));
  context_->SetFuncAttr(std::move(func_attr));
}

} // namespace worker
} // namespace fluidml
