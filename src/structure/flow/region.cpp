#include "structure/flow/region.h"

namespace cpu_transformers {
namespace flow {
Region::Region(std::string &&name, Meta &&meta)
    : name_(std::move(name)), meta_(std::move(meta)) {}

const std::string &Region::GetName() const { return name_; }

const Meta &Region::GetMeta() const { return meta_; }

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
