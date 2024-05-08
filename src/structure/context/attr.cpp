#include "structure/context/attr.h"

namespace cpu_transformers {
namespace context {

ArgumentAttr::ArgumentAttr(Type type, std::string &&name,
                           mlir::MemRefType &&memref_type)
    : type_(type), name_(std::move(name)),
      memref_type_(std::move(memref_type)) {}

ArgumentAttr::Type ArgumentAttr::GetType() const noexcept { return type_; }

const std::string &ArgumentAttr::GetName() const noexcept { return name_; }

const mlir::MemRefType &ArgumentAttr::GetMemRefType() const noexcept {
  return memref_type_;
}

const std::vector<ArgumentAttr> &FuncAttr::GetArguments() const noexcept {
  return arguments_;
}

FuncAttr::FuncAttr(std::string &&name, size_t buffer_size)
    : name_(std::move(name)), buffer_size_(buffer_size) {}

const std::string &FuncAttr::GetName() const noexcept { return name_; }

size_t FuncAttr::GetBuffer() const noexcept { return buffer_size_; }

void FuncAttr::PutArgument(ArgumentAttr &&argument_attr) {
  arguments_.push_back(std::move(argument_attr));
}

} // namespace context
} // namespace cpu_transformers
