#include "structure/flow/edge.h"

namespace cpu_transformers {
namespace flow {
Edge::Edge(std::shared_ptr<Region> &&region) : region_(region) {}

const std::string &Edge::GetName() const { return region_->GetName(); }

const Meta &Edge::GetMeta() const { return region_->GetMeta(); }

const std::string &MemoryEdge::GetFrom() const { return from_; }

const std::string &MemoryEdge::GetTo() const { return to_; }

MemoryEdge::MemoryEdge(std::shared_ptr<Region> &&region, std::string &&from,
                       std::string &&to)
    : Edge(std::move(region)), from_(std::move(from)), to_(std::move(to)) {}

InterfaceEdge::InterfaceEdge(std::shared_ptr<Region> &&region)
    : Edge(std::move(region)) {}

InputEdge::InputEdge(std::shared_ptr<Region> &&region, std::string &&to)
    : InterfaceEdge(std::move(region)), to_(std::move(to)) {}

const std::string &InputEdge::GetTo() const { return to_; }

OutputEdge::OutputEdge(std::shared_ptr<Region> &&region, std::string &&from)
    : InterfaceEdge(std::move(region)), from_(std::move(from)) {}

const std::string &OutputEdge::GetFrom() const { return from_; }
} // namespace flow
} // namespace cpu_transformers
