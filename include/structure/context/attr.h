#ifndef CPU_TRANSFORMERS_STRUCTURE_CONTEXT_ATTR_H_
#define CPU_TRANSFORMERS_STRUCTURE_CONTEXT_ATTR_H_

#include "mlir/IR/BuiltinTypes.h"
#include <string>

namespace cpu_transformers {
namespace context {

class ArgumentAttr {
public:
  enum class Type {
    Input,
    Output,
  };
  ArgumentAttr(Type type, std::string &&name, mlir::MemRefType &&memref_type);
  ArgumentAttr(const ArgumentAttr &argument_attr) = default;
  ArgumentAttr(ArgumentAttr &&argument_attr) = default;
  ArgumentAttr &operator=(const ArgumentAttr &argument_attr) = default;
  ArgumentAttr &operator=(ArgumentAttr &&argument_attr) = default;
  virtual ~ArgumentAttr() = default;
  Type GetType() const noexcept;
  const std::string &GetName() const noexcept;
  const mlir::MemRefType &GetMemRefType() const noexcept;

protected:
  Type type_;
  std::string name_;
  mlir::MemRefType memref_type_;
};

class FuncAttr {
public:
  FuncAttr(std::string &&name, size_t buffer_size);
  FuncAttr(const FuncAttr &func_attr) = default;
  FuncAttr(FuncAttr &&func_attr) = default;
  FuncAttr &operator=(const FuncAttr &func_attr) = default;
  FuncAttr &operator=(FuncAttr &&func_attr) = default;
  virtual ~FuncAttr() = default;
  const std::string &GetName() const noexcept;
  const std::vector<ArgumentAttr> &GetArguments() const noexcept;
  size_t GetBuffer() const noexcept;
  void PutArgument(ArgumentAttr &&argument_attr);

protected:
  std::string name_;
  std::vector<ArgumentAttr> arguments_;
  size_t buffer_size_;
};

} // namespace context
} // namespace cpu_transformers

#endif