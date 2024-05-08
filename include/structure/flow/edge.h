#ifndef CPU_TRANSFORMERS_STRUCTURE_FLOW_EDGE_H_
#define CPU_TRANSFORMERS_STRUCTURE_FLOW_EDGE_H_

#include "structure/flow/region.h"
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
  const std::string &GetName() const;
  const Meta &GetMeta() const;

protected:
  std::shared_ptr<Region> region_;
};

class MemoryEdge : public Edge {
public:
  MemoryEdge(std::shared_ptr<Region> &&region, std::string &&from,
             std::string &&to);
  MemoryEdge(const MemoryEdge &edge) = delete;
  MemoryEdge(MemoryEdge &&edge) = default;
  virtual ~MemoryEdge() = default;
  const std::string &GetFrom() const;
  const std::string &GetTo() const;

private:
  const std::string from_;
  const std::string to_;
};

class InterfaceEdge : public Edge {
public:
  InterfaceEdge(std::shared_ptr<Region> &&region);
  InterfaceEdge(const InterfaceEdge &edge) = delete;
  InterfaceEdge(InterfaceEdge &&edge) = default;
  virtual ~InterfaceEdge() = default;
};

class InputEdge : public InterfaceEdge {
public:
  InputEdge(std::shared_ptr<Region> &&region, std::string &&to);
  InputEdge(const InputEdge &edge) = delete;
  InputEdge(InputEdge &&edge) = default;
  virtual ~InputEdge() = default;
  const std::string &GetTo() const;

private:
  const std::string to_;
};

class OutputEdge : public InterfaceEdge {
public:
  OutputEdge(std::shared_ptr<Region> &&region, std::string &&from);
  OutputEdge(const OutputEdge &edge) = delete;
  OutputEdge(OutputEdge &&edge) = default;
  virtual ~OutputEdge() = default;
  const std::string &GetFrom() const;

private:
  const std::string from_;
};
} // namespace flow
} // namespace cpu_transformers

#endif
