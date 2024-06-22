#ifndef CPU_TRANSFORMERS_STRUCTURE_GRAPH_DEF_H_
#define CPU_TRANSFORMERS_STRUCTURE_GRAPH_DEF_H_

namespace cpu_transformers {
namespace graph {

class Attribute;

class Edge;
class ConstantEdge;
class ConstantScalarEdge;
class ConstantTensorEdge;
class NonConstantEdge;
class PureEdge;
class InputEdge;
class OutputEdge;

class Graph;

class Node;

} // namespace graph
} // namespace cpu_transformers

#endif
