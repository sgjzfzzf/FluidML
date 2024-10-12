#ifndef CPU_TRANSFORMERS_STRUCTURE_FLOW_FWD_H_
#define CPU_TRANSFORMERS_STRUCTURE_FLOW_FWD_H_

namespace cpu_transformers {
namespace flow {

class Edge;
class OwnFromEdge;
class OwnToEdge;
class MemoryEdge;
class InterfaceEdge;
class InputEdge;
class OutputEdge;

class Flow;

class Node;
class SingleInputNode;
class DoubleInputsNode;
class SingleInputWithoutBufferNode;
class SingleInputWithBufferNode;
class DoubleInputsWithoutBufferNode;
class DoubleInputsWithBufferNode;
class AddNode;
class AddConstantNode;
class AddCommonNode;
class AddDivErfAddMulMulNode;
class DivNode;
class DivConstantScalarNode;
class ErfNode;
class GatherNode;
class GatherConstantIndexScalarNode;
class GatherConstantDataTensorNode;
class GatherConstantDataTensorAddTensorLhsAddTensorLhsNode;
class GemmNode;
class GemmConstantWeightsBiasNode;
class LayerNormalizationNode;
class LayerNormalizationConstantScaleBiasNode;
class MatMulNode;
class MulNode;
class MulConstantNode;
class MulCommonNode;
class PowNode;
class ReshapeNode;
class SoftmaxNode;
class SplitNode;
class SubNode;
class SubConstantScalarLhsNode;
class TanhNode;
class TransposeNode;
class UnsqueezeNode;
class UnsqueezeSubLhsScalarMulRhsScalarNode;
class WhereNode;
class WhereConstantCondConstantScalarYNode;
class WhereConstantCondConstantTensorYNode;

class Region;
class InnerRegion;
class InterfaceRegion;
class InputRegion;
class OutputRegion;
class ConstantRegion;

class Sequence;

} // namespace flow
} // namespace cpu_transformers

#endif
