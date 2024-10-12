#ifndef CPU_TRANSFORMERS_STRUCTURE_GRAPH_GRAPH_H_
#define CPU_TRANSFORMERS_STRUCTURE_GRAPH_GRAPH_H_

#include "structure/graph/fwd.h"
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cpu_transformers {
namespace graph {
class Graph {
public:
  Graph() = default;
  Graph(const Graph &graph) = delete;
  Graph(Graph &&graph);
  bool ExistEdge(const std::string &name) const;
  bool PutEdge(std::shared_ptr<Edge> &&edge);
  bool DeleteEdge(const std::string &name);
  bool DeleteEdge(const Edge &edge);
  std::shared_ptr<Edge> GetEdge(const std::string &name) const;
  std::shared_ptr<Node> GetEdgeFrom(const std::string &edge_name) const;
  std::shared_ptr<Node> GetEdgeFrom(const Edge &edge) const;
  std::vector<std::shared_ptr<Node>>
  GetEdgeTo(const std::string &edge_name) const;
  std::vector<std::shared_ptr<Node>> GetEdgeTo(const Edge &edge) const;
  bool ExistNode(const std::string &name) const;
  bool PutNode(std::shared_ptr<Node> &&node);
  bool DeleteNode(const std::string &name);
  bool DeleteNode(const Node &node);
  std::shared_ptr<Node> GetNode(const std::string &name) const;
  std::vector<std::shared_ptr<Edge>>
  GetNodeFrom(const std::string &node_name) const;
  std::vector<std::shared_ptr<Edge>> GetNodeFrom(const Node &node) const;
  std::vector<std::shared_ptr<Edge>>
  GetNodeTo(const std::string &node_name) const;
  std::vector<std::shared_ptr<Edge>> GetNodeTo(const Node &node) const;
  std::vector<std::shared_ptr<Node>>
  GetNextNodes(const std::string &name) const;
  std::vector<std::shared_ptr<Node>> GetNextNodes(const Node &node) const;
  std::vector<std::string> GetNextNodeNames(const std::string &name) const;
  std::vector<std::string> GetNextNodeNames(const Node &node) const;
  std::vector<std::shared_ptr<Edge>> GetAllEdges() const;
  std::vector<std::shared_ptr<InputEdge>> GetInputEdges() const;
  std::vector<std::string> GetInputEdgeNames() const;
  std::vector<std::shared_ptr<Node>> GetAllNodes() const;
  std::vector<std::string> GetAllNodeNames() const;
  bool EdgeToNode(const std::string &edge_name, const std::string &node_name);
  bool EdgeToNode(const Edge &edge, const Node &node);
  bool NodeToEdge(const std::string &node_name, const std::string &edge_name);
  bool NodeToEdge(const Node &node, const Edge &edge);
  bool ClearEdgeToNode(const std::string &edge_name,
                       const std::string &node_name);
  bool ClearEdgeToNode(const Edge &edge, const Node &node);
  bool ClearNodeToEdge(const std::string &node_name,
                       const std::string &edge_name);
  bool ClearNodeToEdge(const Node &node, const Edge &edge);
  bool ClearEdgeFrom(const std::string &edge_name);
  bool ClearEdgeFrom(const Edge &edge);
  bool ClearNodeFrom(const std::string &node_name);
  bool ClearNodeFrom(const Node &node);
  bool ClearEdgeTos(const std::string &edge_name);
  bool ClearEdgeTos(const Edge &edge);
  bool ClearNodeTos(const std::string &node_name);
  bool ClearNodeTos(const Node &node);
  bool Check() const;

private:
  struct EdgeContainer {
    std::shared_ptr<Edge> ptr;
    std::optional<std::string> from;
    std::vector<std::string> to;
  };

  struct NodeContainer {
    std::shared_ptr<Node> ptr;
    std::vector<std::string> from;
    std::vector<std::string> to;
  };
  std::unordered_map<std::string, EdgeContainer> edges_;
  std::unordered_map<std::string, NodeContainer> nodes_;
};
} // namespace graph
} // namespace cpu_transformers

#endif
