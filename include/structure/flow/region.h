#ifndef CPU_TRANSFORMERS_STRUCTURE_FLOW_REGION_H_
#define CPU_TRANSFORMERS_STRUCTURE_FLOW_REGION_H_

#include "structure/flow/object.h"
#include "structure/tensor/meta.h"
#include <string>

namespace cpu_transformers {
namespace flow {
class Region {
public:
  Region(std::string &&name, Meta &&meta);
  Region(std::string &&name, Meta &&meta, std::vector<size_t> &&layout);
  Region(const Region &region) = delete;
  Region(Region &&region) = default;
  virtual ~Region() = default;
  const std::string &GetName() const;
  const Meta &GetMeta() const;
  const std::vector<size_t> &GetLayout() const;
  std::vector<int64_t> GetPhysicalShape() const;
  std::vector<int64_t> GetStrides() const;
  void SetLayout(std::vector<size_t> &&layout);
  virtual bool NeedMemoryAllocation() const = 0;

private:
  std::string name_;
  Meta meta_;
  std::vector<size_t> layout_;
};

class InnerRegion : public Region {
public:
  InnerRegion(std::string &&name, Meta &&meta);
  InnerRegion(const InnerRegion &region) = delete;
  InnerRegion(InnerRegion &&region) = default;
  virtual ~InnerRegion() = default;
  bool NeedMemoryAllocation() const override;
};

class InterfaceRegion : public Region {
public:
  InterfaceRegion(std::string &&name, Meta &&meta);
  InterfaceRegion(const InterfaceRegion &region) = delete;
  InterfaceRegion(InterfaceRegion &&region) = default;
  virtual ~InterfaceRegion() = default;
  bool NeedMemoryAllocation() const override;
};

class InputRegion : public InterfaceRegion {
public:
  InputRegion(std::string &&name, Meta &&meta);
  InputRegion(const InputRegion &region) = delete;
  InputRegion(InputRegion &&region) = default;
  virtual ~InputRegion() = default;
};

class OutputRegion : public InterfaceRegion {
public:
  OutputRegion(std::string &&name, Meta &&meta);
  OutputRegion(const OutputRegion &region) = delete;
  OutputRegion(OutputRegion &&region) = default;
  virtual ~OutputRegion() = default;
};

} // namespace flow
} // namespace cpu_transformers

#endif
