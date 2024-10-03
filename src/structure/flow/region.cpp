#include "structure/flow/region.h"
#include "structure/tensor/tensor.h"
#include "utils/utils.h"

namespace cpu_transformers {
namespace flow {

Region::Region(std::string &&name, std::vector<size_t> &&layout)
    : name_(std::move(name)), layout_(std::move(layout)) {}

const std::string &Region::GetName() const { return name_; }

const std::vector<size_t> &Region::GetLayout() const { return layout_; }

std::vector<int64_t> Region::GetPhysicalShape() const {
  const Meta &meta = GetMeta();
  const std::vector<int64_t> &shape = meta.GetShape();
  return utils::GenPhysicalShape(shape, layout_);
}

std::vector<int64_t> Region::GetStrides() const {
  const Meta &meta = GetMeta();
  const std::vector<int64_t> &shape = meta.GetShape();
  return utils::GenStrides(shape, layout_);
}

void Region::SetLayout(std::vector<size_t> &&layout) { layout_ = layout; }

std::vector<int64_t> Region::getDefaultLayout() const {
  const Meta &meta = GetMeta();
  const std::vector<int64_t> &shape = meta.GetShape();
  const size_t shape_len = shape.size();
  std::vector<int64_t> layout(shape_len);
  for (size_t i = 0; i < shape_len; ++i) {
    layout[i] = i;
  }
  return layout;
}

VariableRegion::VariableRegion(std::string &&name, Meta &&meta)
    : Region(std::move(name), utils::GenDefaultLayout(meta.GetShape())),
      meta_(std::move(meta)) {}

VariableRegion::VariableRegion(std::string &&name, Meta &&meta,
                               std::vector<size_t> &&layout)
    : Region(std::move(name), std::move(layout)), meta_(std::move(meta)) {}

const Meta &VariableRegion::GetMeta() const { return meta_; }

InnerRegion::InnerRegion(std::string &&name, Meta &&meta)
    : VariableRegion(std::move(name), std::move(meta)) {}

InnerRegion::InnerRegion(std::string &&name, Meta &&meta,
                         std::vector<size_t> &&layout)
    : VariableRegion(std::move(name), std::move(meta), std::move(layout)) {}

bool InnerRegion::NeedMemoryAllocation() const { return true; }

InterfaceRegion::InterfaceRegion(std::string &&name, Meta &&meta)
    : VariableRegion(std::move(name), std::move(meta)) {}

InterfaceRegion::InterfaceRegion(std::string &&name, Meta &&meta,
                                 std::vector<size_t> &&layout)
    : VariableRegion(std::move(name), std::move(meta), std::move(layout)) {}

bool InterfaceRegion::NeedMemoryAllocation() const { return false; }

InputRegion::InputRegion(std::string &&name, Meta &&meta)
    : InterfaceRegion(std::move(name), std::move(meta)) {}

InputRegion::InputRegion(std::string &&name, Meta &&meta,
                         std::vector<size_t> &&layout)
    : InterfaceRegion(std::move(name), std::move(meta), std::move(layout)) {}

OutputRegion::OutputRegion(std::string &&name, Meta &&meta)
    : InterfaceRegion(std::move(name), std::move(meta)) {}

OutputRegion::OutputRegion(std::string &&name, Meta &&meta,
                           std::vector<size_t> &&layout)
    : InterfaceRegion(std::move(name), std::move(meta), std::move(layout)) {}

ConstantRegion::ConstantRegion(std::string &&name, Tensor &&tensor)
    : Region(std::move(name), utils::GenDefaultLayout(tensor.GetShape())),
      constant_(std::move(tensor)) {}

ConstantRegion::ConstantRegion(std::string &&name, Tensor &&tensor,
                               std::vector<size_t> &&layout)
    : Region(std::move(name), std::move(layout)), constant_(std::move(tensor)) {
}

const Meta &ConstantRegion::GetMeta() const { return constant_.GetMeta(); }

bool ConstantRegion::NeedMemoryAllocation() const { return false; }

} // namespace flow
} // namespace cpu_transformers
