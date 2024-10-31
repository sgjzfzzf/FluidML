#ifndef FLUIDML_STRUCTURE_FLOW_FWD_H_
#define FLUIDML_STRUCTURE_FLOW_FWD_H_

namespace fluidml {
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
class CastNode;
class ConcatNode;
class Concat2CommonNode;
class ConvNode;
class ConvWithoutPaddingNode;
class ConvWithPaddingNode;
class CumSumNode;
class DivNode;
class DivConstantRhsNode;
class DivCommonNode;
class DropoutNode;
class EqualNode;
class ErfNode;
class FlattenNode;
class GatherNode;
class GatherConstantIndexScalarNode;
class GatherConstantIndicesTensorNode;
class GatherConstantDataTensorNode;
class GatherConstantDataTensorAddTensorLhsAddTensorLhsNode;
class GemmNode;
class GemmConstantWeightsBiasNode;
class LayerNormalizationNode;
class LayerNormalizationConstantScaleBiasNode;
class MatMulNode;
class MaxPoolNode;
class MaxPoolWithoutPaddingNode;
class MulNode;
class MulConstantNode;
class MulCommonNode;
class NegNode;
class NotNode;
class PadNode;
class PowNode;
class ReduceMeanNode;
class ReluNode;
class ReshapeNode;
class SliceNode;
class SoftmaxNode;
class SqrtNode;
class SubNode;
class SubConstantLhsNode;
class SubCommonNode;
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
} // namespace fluidml

#endif
