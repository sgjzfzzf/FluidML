#ifndef CPU_TRANSFORMERS_STRUCTURE_FLOW_EDGE_H_
#define CPU_TRANSFORMERS_STRUCTURE_FLOW_EDGE_H_

#include "structure/flow/fwd.h"
#include "structure/tensor/meta.h"
#include <string>

namespace cpu_transformers {
namespace flow {

class Edge {
public:
  Edge(std::shared_ptr<Region> &&region);
  Edge(const Edge &edge) = delete;
  Edge(Edge &&edge) = default;
  virtual ~Edge() = default;
  std::shared_ptr<Region> GetRegion();
  const std::string &GetName() const;
  const Meta &GetMeta() const;
  const std::vector<size_t> &GetLayout() const;
  std::vector<int64_t> GePhysicalShape() const;

protected:
  std::shared_ptr<Region> region_;
};

class OwnFromEdge : virtual public Edge {
public:
  OwnFromEdge(std::shared_ptr<Region> &&region, std::shared_ptr<Node> &&from);
  OwnFromEdge(const OwnFromEdge &edge) = delete;
  OwnFromEdge(OwnFromEdge &&edge) = default;
  virtual ~OwnFromEdge() = default;
  std::shared_ptr<Node> GetFrom() const;

protected:
  std::shared_ptr<Node> from_;
};

class OwnToEdge : virtual public Edge {
public:
  OwnToEdge(std::shared_ptr<Region> &&region, std::shared_ptr<Node> &&to);
  OwnToEdge(const OwnToEdge &edge) = delete;
  OwnToEdge(OwnToEdge &&edge) = default;
  virtual ~OwnToEdge() = default;
  std::shared_ptr<Node> GetTo() const;

protected:
  std::shared_ptr<Node> to_;
};

class MemoryEdge : public OwnFromEdge, public OwnToEdge {
public:
  MemoryEdge(std::shared_ptr<Region> &&region, std::shared_ptr<Node> &&from,
             std::shared_ptr<Node> &&to);
  MemoryEdge(const MemoryEdge &edge) = delete;
  MemoryEdge(MemoryEdge &&edge) = default;
  virtual ~MemoryEdge() = default;
};

class InterfaceEdge : virtual public Edge {
public:
  InterfaceEdge(std::shared_ptr<Region> &&region);
  InterfaceEdge(const InterfaceEdge &edge) = delete;
  InterfaceEdge(InterfaceEdge &&edge) = default;
  virtual ~InterfaceEdge() = default;
};

class InputEdge : public InterfaceEdge, public OwnToEdge {
public:
  InputEdge(std::shared_ptr<Region> &&region, std::shared_ptr<Node> &&to);
  InputEdge(const InputEdge &edge) = delete;
  InputEdge(InputEdge &&edge) = default;
  virtual ~InputEdge() = default;
};

class OutputEdge : public InterfaceEdge, public OwnFromEdge {
public:
  OutputEdge(std::shared_ptr<Region> &&region, std::shared_ptr<Node> &&from);
  OutputEdge(const OutputEdge &edge) = delete;
  OutputEdge(OutputEdge &&edge) = default;
  virtual ~OutputEdge() = default;
};

class ConstantEdge : public OwnToEdge {
public:
  ConstantEdge(std::shared_ptr<Region> &&region, std::shared_ptr<Node> &&to);
  ConstantEdge(const ConstantEdge &edge) = delete;
  ConstantEdge(ConstantEdge &&edge) = default;
  virtual ~ConstantEdge() = default;
};

} // namespace flow
} // namespace cpu_transformers

#endif
