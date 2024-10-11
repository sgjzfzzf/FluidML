#ifndef CPU_TRANSFORMERS_STRUCTURE_FLOW_REGION_H_
#define CPU_TRANSFORMERS_STRUCTURE_FLOW_REGION_H_

#include "structure/flow/object.h"
#include "structure/tensor/meta.h"
#include "structure/tensor/tensor.h"
#include "utils/float.h"
#include <string>

namespace cpu_transformers {
namespace flow {
class Region {
public:
  Region(std::string &&name, std::vector<size_t> &&layout);
  Region(const Region &region) = delete;
  Region(Region &&region) = default;
  virtual ~Region() = default;
  const std::string &GetName() const;
  virtual const Meta &GetMeta() const = 0;
  const std::vector<size_t> &GetLayout() const;
  std::vector<int64_t> GetPhysicalShape() const;
  std::vector<int64_t> GetStrides() const;
  void SetLayout(std::vector<size_t> &&layout);
  virtual bool NeedMemoryAllocation() const = 0;

protected:
  std::vector<int64_t> getDefaultLayout() const;
  std::string name_;
  std::vector<size_t> layout_;
};

class VariableRegion : public Region {
public:
  VariableRegion(std::string &&name, Meta &&meta);
  VariableRegion(std::string &&name, Meta &&meta, std::vector<size_t> &&layout);
  VariableRegion(const VariableRegion &region) = delete;
  VariableRegion(VariableRegion &&region) = default;
  virtual ~VariableRegion() = default;
  const Meta &GetMeta() const override;
  bool NeedMemoryAllocation() const override = 0;

protected:
  Meta meta_;
};

class InnerRegion : public VariableRegion {
public:
  InnerRegion(std::string &&name, Meta &&meta);
  InnerRegion(std::string &&name, Meta &&meta, std::vector<size_t> &&layout);
  InnerRegion(const InnerRegion &region) = delete;
  InnerRegion(InnerRegion &&region) = default;
  virtual ~InnerRegion() = default;
  bool NeedMemoryAllocation() const override;
};

class InterfaceRegion : public VariableRegion {
public:
  InterfaceRegion(std::string &&name, Meta &&meta);
  InterfaceRegion(std::string &&name, Meta &&meta,
                  std::vector<size_t> &&layout);
  InterfaceRegion(const InterfaceRegion &region) = delete;
  InterfaceRegion(InterfaceRegion &&region) = default;
  virtual ~InterfaceRegion() = default;
  bool NeedMemoryAllocation() const override;
};

class InputRegion : public InterfaceRegion {
public:
  InputRegion(std::string &&name, Meta &&meta);
  InputRegion(std::string &&name, Meta &&meta, std::vector<size_t> &&layout);
  InputRegion(const InputRegion &region) = delete;
  InputRegion(InputRegion &&region) = default;
  virtual ~InputRegion() = default;
};

class OutputRegion : public InterfaceRegion {
public:
  OutputRegion(std::string &&name, Meta &&meta);
  OutputRegion(std::string &&name, Meta &&meta, std::vector<size_t> &&layout);
  OutputRegion(const OutputRegion &region) = delete;
  OutputRegion(OutputRegion &&region) = default;
  virtual ~OutputRegion() = default;
};

class ConstantRegion : public Region {
public:
  ConstantRegion(std::string &&name, Tensor &&tensor);
  ConstantRegion(std::string &&name, Tensor &&tensor,
                 std::vector<size_t> &&layout);
  ConstantRegion(const ConstantRegion &region) = delete;
  ConstantRegion(ConstantRegion &&region) = default;
  virtual ~ConstantRegion() = default;
  const Meta &GetMeta() const override;
  const Tensor &GetTensor() const;
  bool NeedMemoryAllocation() const override;

private:
  Tensor tensor_;
};

} // namespace flow
} // namespace cpu_transformers

#endif
