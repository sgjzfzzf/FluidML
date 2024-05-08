#include "structure/graph/edge.h"
#include "structure/graph/graph.h"
#include "structure/graph/node.h"
#include "utils/type.h"
#include "gtest/gtest.h"
#include <memory>

TEST(GraphTest, BasicTest) {
  using namespace cpu_transformers;
  using namespace cpu_transformers::graph;

  std::unique_ptr<Graph> graph = std::make_unique<Graph>();
  ASSERT_TRUE(graph->AddEdge(std::make_shared<PureEdge>(
      "edge0", Type::FLOAT32, std::vector<int64_t>{1, 2, 3})));
  ASSERT_TRUE(graph->AddEdge(std::make_shared<PureEdge>(
      "edge1", Type::FLOAT32, std::vector<int64_t>{1, 2})));
  ASSERT_TRUE(graph->AddEdge(std::make_shared<PureEdge>(
      "edge2", Type::FLOAT32, std::vector<int64_t>{1})));
  ASSERT_TRUE(graph->ExistEdge("edge0"));
  ASSERT_TRUE(graph->ExistEdge("edge1"));
  ASSERT_TRUE(graph->ExistEdge("edge2"));
  ASSERT_TRUE(graph->PutNode(std::make_shared<Node>("node0", Node::Op::Add)));
  ASSERT_TRUE(
      graph->PutNode(std::make_shared<Node>("node1", Node::Op::MatMul)));
  ASSERT_TRUE(graph->ExistNode("node0"));
  ASSERT_TRUE(graph->ExistNode("node1"));
  ASSERT_TRUE(graph->EdgeToNode("edge0", "node0"));
  ASSERT_TRUE(graph->NodeToEdge("node0", "edge1"));
  ASSERT_TRUE(graph->EdgeToNode("edge1", "node1"));
  ASSERT_TRUE(graph->NodeToEdge("node1", "edge2"));
  std::vector<std::shared_ptr<Node>> node_from_edge0 =
      graph->GetEdgeTo("edge0");
  ASSERT_EQ(node_from_edge0.size(), 1);
  ASSERT_EQ(node_from_edge0.front()->GetName(), "node0");
  std::vector<std::shared_ptr<Edge>> edge_to_node0 = graph->GetNodeTo("node0");
  ASSERT_EQ(edge_to_node0.size(), 1);
  std::vector<std::shared_ptr<Edge>> edge_from_node0 =
      graph->GetNodeFrom("node0");
  ASSERT_EQ(edge_from_node0.size(), 1);
  ASSERT_EQ(edge_from_node0.front()->GetName(), "edge0");
  std::vector<std::shared_ptr<Node>> node_from_edge1 =
      graph->GetEdgeTo("edge1");
  ASSERT_EQ(node_from_edge1.size(), 1);
  ASSERT_EQ(node_from_edge1.front()->GetName(), "node1");
  std::vector<std::shared_ptr<Edge>> edge_to_node1 = graph->GetNodeTo("node1");
  ASSERT_EQ(edge_to_node1.size(), 1);
  std::optional<std::shared_ptr<Node>> node_to_edge2 =
      graph->GetEdgeFrom("edge2");
  ASSERT_TRUE(node_to_edge2.has_value());
  ASSERT_EQ((*node_to_edge2)->GetName(), "node1");
  std::vector<std::shared_ptr<Node>> next_nodes = graph->GetNextNodes("node0");
  ASSERT_EQ(next_nodes.size(), 1);
  ASSERT_EQ(next_nodes.front()->GetName(), "node1");
  ASSERT_TRUE(graph->DeleteEdge("edge0"));
  ASSERT_FALSE(graph->ExistEdge("edge0"));
  ASSERT_TRUE(graph->ExistEdge("edge1"));
  ASSERT_TRUE(graph->ExistEdge("edge2"));
}
