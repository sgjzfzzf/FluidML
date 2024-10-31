#include "structure/flow/edge.h"
#include "structure/flow/region.h"
#include <memory>

namespace fluidml {
namespace flow {

Edge::Edge(std::shared_ptr<Region> &&region) : region_(region) {}

std::shared_ptr<Region> Edge::GetRegion() { return region_; }

const std::string &Edge::GetName() const { return region_->GetName(); }

const Meta &Edge::GetMeta() const { return region_->GetMeta(); }

const std::vector<size_t> &Edge::GetLayout() const {
  return region_->GetLayout();
}

std::vector<int64_t> Edge::GePhysicalShape() const {
  return region_->GetPhysicalShape();
}

OwnFromEdge::OwnFromEdge(std::shared_ptr<Region> &&region,
                         std::shared_ptr<Node> &&from)
    : Edge(std::move(region)), from_(std::move(from)) {}

std::shared_ptr<Node> OwnFromEdge::GetFrom() const { return from_; }

OwnToEdge::OwnToEdge(std::shared_ptr<Region> &&region,
                     std::shared_ptr<Node> &&to)
    : Edge(std::move(region)), to_(std::move(to)) {}

std::shared_ptr<Node> OwnToEdge::GetTo() const { return to_; }

MemoryEdge::MemoryEdge(std::shared_ptr<Region> &&region,
                       std::shared_ptr<Node> &&from, std::shared_ptr<Node> &&to)
    : Edge(std::move(region)), OwnFromEdge(std::move(region), std::move(from)),
      OwnToEdge(std::move(region), std::move(to)) {}

InterfaceEdge::InterfaceEdge(std::shared_ptr<Region> &&region)
    : Edge(std::move(region)) {}

InputEdge::InputEdge(std::shared_ptr<Region> &&region,
                     std::shared_ptr<Node> &&to)
    : Edge(std::move(region)), InterfaceEdge(std::move(region)),
      OwnToEdge(std::move(region), std::move(to)) {}

OutputEdge::OutputEdge(std::shared_ptr<Region> &&region,
                       std::shared_ptr<Node> &&from)
    : Edge(std::move(region)), InterfaceEdge(std::move(region)),
      OwnFromEdge(std::move(region), std::move(from)) {}

ConstantEdge::ConstantEdge(std::shared_ptr<Region> &&region,
                           std::shared_ptr<Node> &&to)
    : Edge(std::move(region)), OwnToEdge(std::move(region), std::move(to)) {}

} // namespace flow
} // namespace fluidml
