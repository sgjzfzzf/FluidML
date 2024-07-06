#include "structure/flow/region.h"
#include "evaluation/utils.h"

namespace cpu_transformers {
namespace flow {

Region::Region(std::string &&name, Meta &&meta)
    : name_(std::move(name)), meta_(std::move(meta)) {
  const std::vector<int64_t> &shape = meta_.GetShape();
  const size_t shape_len = shape.size();
  layout_.resize(shape_len);
  for (size_t i = 0; i < shape_len; ++i) {
    layout_[i] = i;
  }
}

Region::Region(std::string &&name, Meta &&meta, std::vector<size_t> &&layout)
    : name_(std::move(name)), meta_(std::move(meta)),
      layout_(std::move(layout)) {}

const std::string &Region::GetName() const { return name_; }

const Meta &Region::GetMeta() const { return meta_; }

const std::vector<size_t> &Region::GetLayout() const { return layout_; }

std::vector<int64_t> Region::GetPhysicalShape() const {
  const std::vector<int64_t> &shape = meta_.GetShape();
  return evaluation::GenPhysicalShape(shape, layout_);
}

std::vector<int64_t> Region::GetStrides() const {
  const std::vector<int64_t> &shape = meta_.GetShape();
  return evaluation::GenStrides(shape, layout_);
}

void Region::SetLayout(std::vector<size_t> &&layout) { layout_ = layout; }

InnerRegion::InnerRegion(std::string &&name, Meta &&meta)
    : Region(std::move(name), std::move(meta)) {}

bool InnerRegion::NeedMemoryAllocation() const { return true; }

InterfaceRegion::InterfaceRegion(std::string &&name, Meta &&meta)
    : Region(std::move(name), std::move(meta)) {}

bool InterfaceRegion::NeedMemoryAllocation() const { return false; }

InputRegion::InputRegion(std::string &&name, Meta &&meta)
    : InterfaceRegion(std::move(name), std::move(meta)) {}

OutputRegion::OutputRegion(std::string &&name, Meta &&meta)
    : InterfaceRegion(std::move(name), std::move(meta)) {}

} // namespace flow
} // namespace cpu_transformers
