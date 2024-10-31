import argparse
import json
import onnx

from typing import Dict, List, Tuple, Union


class Analyzer(object):
    def __init__(
        self,
        dp_table: Dict[str, Dict[str, List[int]]],
        plan: Dict[str, List[int]],
        infos: Dict[str, List[int]],
    ):
        self.dp_table: Dict[str, List[int]] = dp_table
        self.plan: Dict[str, List[int]] = plan
        self.infos: Dict[str, List[int]] = infos

    def find_single_input(
        self,
        name: str,
        input: str,
        output: str,
    ) -> Tuple[int, int]:
        input_shape: List[int] = self.infos[input]
        output_shape: List[int] = self.infos[output]
        default_input_layout: List[int] = [i for i, _ in enumerate(input_shape)]
        default_output_layout: List[int] = [i for i, _ in enumerate(output_shape)]
        input_layout: List[int] = self.plan[input]
        output_layout: List[int] = self.plan[output]
        default_item: Dict[str, Union[List[int], int]] = next(
            filter(
                lambda x: x["input_shape"] == input_shape
                and x["output_shape"] == output_shape
                and x["input_layout"] == default_input_layout
                and x["output_layout"] == default_output_layout,
                self.dp_table[name],
            )
        )
        item: Dict[str, Union[List[int], int]] = next(
            filter(
                lambda x: x["input_shape"] == input_shape
                and x["output_shape"] == output_shape
                and x["input_layout"] == input_layout
                and x["output_layout"] == output_layout,
                self.dp_table[name],
            )
        )
        default_time_cost: int = default_item["time_cost"]
        time_cost: int = item["time_cost"]
        return default_time_cost, time_cost

    def find_double_inputs(
        self,
        name: str,
        lhs: str,
        rhs: str,
        output: str,
    ) -> Tuple[int, int]:
        lhs_shape: List[int] = self.infos[lhs]
        rhs_shape: List[int] = self.infos[rhs]
        output_shape: List[int] = self.infos[output]
        default_lhs_layout: List[int] = [i for i, _ in enumerate(lhs_shape)]
        default_rhs_layout: List[int] = [i for i, _ in enumerate(rhs_shape)]
        default_output_layout: List[int] = [i for i, _ in enumerate(output_shape)]
        lhs_layout: List[int] = self.plan[lhs]
        rhs_layout: List[int] = self.plan[rhs]
        output_layout: List[int] = self.plan[output]
        default_item: Dict[str, Union[List[int], int]] = next(
            filter(
                lambda x: x["lhs_shape"] == lhs_shape
                and x["rhs_shape"] == rhs_shape
                and x["output_shape"] == output_shape
                and x["lhs_layout"] == default_lhs_layout
                and x["rhs_layout"] == default_rhs_layout
                and x["output_layout"] == default_output_layout,
                self.dp_table[name],
            )
        )
        item: Dict[str, Union[List[int], int]] = next(
            filter(
                lambda x: x["lhs_shape"] == lhs_shape
                and x["rhs_shape"] == rhs_shape
                and x["output_shape"] == output_shape
                and x["lhs_layout"] == lhs_layout
                and x["rhs_layout"] == rhs_layout
                and x["output_layout"] == output_layout,
                self.dp_table[name],
            )
        )
        default_time_cost: int = default_item["time_cost"]
        time_cost: int = item["time_cost"]
        return default_time_cost, time_cost


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Analyze the compiler execution intermediate results."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The ONNX model file.",
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="The intermediate json file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output json file.",
    )
    args: argparse.Namespace = parser.parse_args()
    model: onnx.ModelProto = onnx.load(args.model)
    with open(args.json, "r") as f:
        data: Dict = json.load(f)
    dp_table: Dict[str, List[int]] = data["dp_table"]["evaluator"]
    plan: Dict[str, List[int]] = data["plan"]
    value_infos: Dict[str, List[int]] = (
        {
            input.name: [dim.dim_value for dim in input.type.tensor_type.shape.dim]
            for input in model.graph.input
        }
        | {
            output.name: [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            for output in model.graph.output
        }
        | {
            value_info.name: [
                dim.dim_value for dim in value_info.type.tensor_type.shape.dim
            ]
            for value_info in model.graph.value_info
        }
    )
    initializers: Dict[str, List[int]] = {
        initializer.name: [dim for dim in initializer.dims]
        for initializer in model.graph.initializer
    }
    analyzer: Analyzer = Analyzer(dp_table, plan, value_infos | initializers)
    table: Dict[str, Tuple[int, int]] = {name: (0, 0) for name in dp_table}
    for node in model.graph.node:
        if node.op_type == "Add":
            assert len(node.input) == 2 and any(
                map(
                    lambda input: input in value_infos,
                    node.input,
                )
            )
            if any(
                map(
                    lambda input: input in initializers
                    and len(initializers[input]) == 0,
                    node.input,
                )
            ):
                input: str = next(
                    filter(
                        lambda input: input in value_infos,
                        node.input,
                    )
                )
                assert input in value_infos
                [output] = node.output
                default_time_cost, time_cost = analyzer.find_single_input(
                    "AddConstantKernel", input, output
                )
                m, n = table["AddConstantKernel"]
                table["AddConstantKernel"] = (m + default_time_cost, n + time_cost)
            else:
                [lhs, rhs] = node.input
                [output] = node.output
                default_time_cost, time_cost = analyzer.find_double_inputs(
                    "AddCommonKernel", lhs, rhs, output
                )
                m, n = table["AddCommonKernel"]
                table["AddCommonKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Cast":
            assert len(node.input) == 1 and all(
                map(lambda input: input in value_infos, node.input)
            )
            [input] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "CastKernel", input, output
            )
            m, n = table["CastKernel"]
            table["CastKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Concat":
            assert len(node.input) == 2 and all(
                map(lambda input: input in value_infos, node.input)
            )
            [lhs, rhs] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_double_inputs(
                "Concat2Kernel", lhs, rhs, output
            )
            m, n = table["Concat2Kernel"]
            table["Concat2Kernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Conv":
            if all(
                map(
                    lambda attr: all(map(lambda elem: elem == 0, attr.ints)),
                    (filter(lambda attr: attr.name == "pads", node.attribute)),
                )
            ):
                name: str = "ConvWithoutPaddingKernel"
            else:
                name: str = "ConvWithPaddingKernel"
            if len(node.input) == 2:
                [input, weights] = node.input
            elif len(node.input) == 3:
                [input, weights, _] = node.input
            else:
                assert False, "unimplemented"
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_double_inputs(
                name, input, weights, output
            )
            m, n = table[name]
            table[name] = (
                m + default_time_cost,
                n + time_cost,
            )
        elif node.op_type == "CumSum":
            assert (
                len(node.input) == 2
                and node.input[0] in value_infos
                and node.input[1] in initializers
            )
            [input, _] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "CumSumKernel", input, output
            )
            m, n = table["CumSumKernel"]
            table["CumSumKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Div":
            assert len(node.input) == 2 and node.input[0] in value_infos
            if node.input[1] in initializers and len(initializers[node.input[1]]) == 0:
                [input, _] = node.input
                [output] = node.output
                default_time_cost, time_cost = analyzer.find_single_input(
                    "DivConstantRhsKernel", input, output
                )
                m, n = table["DivConstantRhsKernel"]
                table["DivConstantRhsKernel"] = (m + default_time_cost, n + time_cost)
            elif (
                node.input[1] in value_infos
                or node.input[1] in initializers
                and len(initializers[node.input[1]]) > 0
            ):
                [lhs, rhs] = node.input
                [output] = node.output
                default_time_cost, time_cost = analyzer.find_double_inputs(
                    "DivCommonKernel", lhs, rhs, output
                )
                m, n = table["DivCommonKernel"]
                table["DivCommonKernel"] = (m + default_time_cost, n + time_cost)
            else:
                assert False
        elif node.op_type == "Dropout":
            assert (
                len(node.input) == 2
                and node.input[0] in value_infos
                and node.input[1] in initializers
            )
            [input, _] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "DropoutKernel", input, output
            )
            m, n = table["DropoutKernel"]
            table["DropoutKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Equal":
            assert len(node.input) == 2 and node.input[0] in value_infos
            [input, _] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "EqualKernel", input, output
            )
            m, n = table["EqualKernel"]
            table["EqualKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Erf":
            assert len(node.input) == 1 and all(
                map(lambda input: input in value_infos, node.input)
            )
            [input] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "ErfKernel", input, output
            )
            m, n = table["ErfKernel"]
            table["ErfKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Flatten":
            assert len(node.input) == 1 and all(
                map(lambda input: input in value_infos, node.input)
            )
            [input] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "FlattenKernel", input, output
            )
            m, n = table["FlattenKernel"]
            table["FlattenKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Gather":
            assert len(node.input) == 2
            if node.input[0] in initializers and node.input[1] in value_infos:
                [_, input] = node.input
                [output] = node.output
                default_time_cost, time_cost = analyzer.find_single_input(
                    "GatherConstantDataTensorKernel", input, output
                )
                m, n = table["GatherConstantDataTensorKernel"]
                table["GatherConstantDataTensorKernel"] = (
                    m + default_time_cost,
                    n + time_cost,
                )
            elif node.input[0] in value_infos and node.input[1] in initializers:
                indices = initializers[node.input[1]]
                if len(indices) == 0:
                    name: str = "GatherConstantIndexScalarKernel"
                else:
                    name: str = "GatherConstantIndicesTensorKernel"
                [input, _] = node.input
                [output] = node.output
                default_time_cost, time_cost = analyzer.find_single_input(
                    name, input, output
                )
                m, n = table[name]
                table[name] = (
                    m + default_time_cost,
                    n + time_cost,
                )
            else:
                assert False, "unimplemented"
        elif node.op_type == "Gemm":
            assert (
                len(node.input) == 3
                and node.input[0] in value_infos
                and all(map(lambda input: input in initializers, node.input[1:]))
            )
            [lhs, rhs, _] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_double_inputs(
                "GemmConstantWeightsBiasKernel", lhs, rhs, output
            )
            m, n = table["GemmConstantWeightsBiasKernel"]
            table["GemmConstantWeightsBiasKernel"] = (
                m + default_time_cost,
                n + time_cost,
            )
        elif node.op_type == "LayerNormalization":
            if (
                len(node.input) == 3
                and node.input[0] in value_infos
                and all(map(lambda input: input in initializers, node.input[1:]))
            ):
                [input, _, _] = node.input
                [output] = node.output
                default_time_cost, time_cost = analyzer.find_single_input(
                    "LayerNormalizationConstantScaleBiasKernel", input, output
                )
                m, n = table["LayerNormalizationConstantScaleBiasKernel"]
                table["LayerNormalizationConstantScaleBiasKernel"] = (
                    m + default_time_cost,
                    n + time_cost,
                )
            else:
                assert False, "unimplemented"
        elif node.op_type == "MatMul":
            assert len(node.input) == 2
            [lhs, rhs] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_double_inputs(
                "MatMulKernel", lhs, rhs, output
            )
            m, n = table["MatMulKernel"]
            table["MatMulKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "MaxPool":
            if all(
                map(
                    lambda attr: all(map(lambda elem: elem == 0, attr.ints)),
                    (filter(lambda attr: attr.name == "pads", node.attribute)),
                )
            ):
                name: str = "MaxPoolWithoutPaddingKernel"
            else:
                name: str = "MaxPoolWithPaddingKernel"
            [input] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                name, input, output
            )
            m, n = table[name]
            table[name] = (
                m + default_time_cost,
                n + time_cost,
            )
        elif node.op_type == "Mul":
            assert len(node.input) == 2 and any(
                map(
                    lambda input: input in value_infos,
                    node.input,
                )
            )
            if any(
                map(
                    lambda input: input in initializers
                    and len(initializers[input]) == 0,
                    node.input,
                )
            ):
                input: str = next(
                    filter(
                        lambda input: input in value_infos,
                        node.input,
                    )
                )
                assert input in value_infos
                [output] = node.output
                default_time_cost, time_cost = analyzer.find_single_input(
                    "MulConstantKernel", input, output
                )
                m, n = table["MulConstantKernel"]
                table["MulConstantKernel"] = (m + default_time_cost, n + time_cost)
            else:
                [lhs, rhs] = node.input
                [output] = node.output
                default_time_cost, time_cost = analyzer.find_double_inputs(
                    "MulCommonKernel", lhs, rhs, output
                )
                m, n = table["MulCommonKernel"]
                table["MulCommonKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Neg":
            assert len(node.input) == 1 and all(
                map(lambda input: input in value_infos, node.input)
            )
            [input] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "NegKernel", input, output
            )
            m, n = table["NegKernel"]
            table["NegKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Not":
            assert len(node.input) == 1 and all(
                map(lambda input: input in value_infos, node.input)
            )
            [input] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "NotKernel", input, output
            )
            m, n = table["NotKernel"]
            table["NotKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Pad":
            assert (
                len(node.input) == 2
                and node.input[0] in value_infos
                and node.input[1] in initializers
            )
            [input, _] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "PadKernel", input, output
            )
            m, n = table["PadKernel"]
            table["PadKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Pow":
            assert (
                len(node.input) == 2
                and node.input[0] in value_infos
                and node.input[1] in initializers
            )
            [input, _] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "PowKernel", input, output
            )
            m, n = table["PowKernel"]
            table["PowKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "ReduceMean":
            assert len(node.input) == 1 and all(
                map(lambda input: input in value_infos, node.input)
            )
            [input] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "ReduceMeanKernel", input, output
            )
            m, n = table["ReduceMeanKernel"]
            table["ReduceMeanKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Relu":
            assert len(node.input) == 1 and all(
                map(lambda input: input in value_infos, node.input)
            )
            [input] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "ReluKernel", input, output
            )
            m, n = table["ReluKernel"]
            table["ReluKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Reshape":
            assert (
                len(node.input) == 2
                and node.input[0] in value_infos
                and node.input[1] in initializers
            )
            [input, _] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "ReshapeKernel", input, output
            )
            m, n = table["ReshapeKernel"]
            table["ReshapeKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Slice":
            assert (
                len(node.input) == 5
                and node.input[0] in value_infos
                and all(map(lambda input: input in initializers, node.input[1:]))
            )
            [input, _, _, _, _] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "SliceKernel", input, output
            )
            m, n = table["SliceKernel"]
            table["SliceKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Softmax":
            assert len(node.input) == 1 and all(
                map(lambda input: input in value_infos, node.input)
            )
            [input] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "SoftmaxKernel", input, output
            )
            m, n = table["SoftmaxKernel"]
            table["SoftmaxKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Sqrt":
            assert len(node.input) == 1 and all(
                map(lambda input: input in value_infos, node.input)
            )
            [input] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "SqrtKernel", input, output
            )
            m, n = table["SqrtKernel"]
            table["SqrtKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Sub":
            assert len(node.input) == 2 and any(
                map(
                    lambda input: input in value_infos,
                    node.input,
                )
            )
            if node.input[0] in initializers and node.input[1] in value_infos:
                [_, input] = node.input
                [output] = node.output
                default_time_cost, time_cost = analyzer.find_single_input(
                    "SubConstantLhsKernel", input, output
                )
                m, n = table["SubConstantLhsKernel"]
                table["SubConstantLhsKernel"] = (m + default_time_cost, n + time_cost)
            elif all(
                map(
                    lambda input: input in value_infos,
                    node.input,
                )
            ):
                [lhs, rhs] = node.input
                [output] = node.output
                default_time_cost, time_cost = analyzer.find_double_inputs(
                    "SubCommonKernel", lhs, rhs, output
                )
                m, n = table["SubCommonKernel"]
                table["SubCommonKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Tanh":
            assert len(node.input) == 1 and all(
                map(lambda input: input in value_infos, node.input)
            )
            [input] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "TanhKernel", input, output
            )
            m, n = table["TanhKernel"]
            table["TanhKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Transpose":
            assert len(node.input) == 1 and all(
                map(lambda input: input in value_infos, node.input)
            )
            [input] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "TransposeKernel", input, output
            )
            m, n = table["TransposeKernel"]
            table["TransposeKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Unsqueeze":
            assert (
                len(node.input) == 2
                and node.input[0] in value_infos
                and node.input[1] in initializers
            )
            [input, _] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "UnsqueezeKernel", input, output
            )
            m, n = table["UnsqueezeKernel"]
            table["UnsqueezeKernel"] = (m + default_time_cost, n + time_cost)
        elif node.op_type == "Where":
            assert (
                len(node.input) == 3
                and node.input[0] in initializers
                and node.input[1] in value_infos
                and node.input[2] in initializers
            )
            [_, input, _] = node.input
            [output] = node.output
            default_time_cost, time_cost = analyzer.find_single_input(
                "WhereConstantCondConstantScalarYKernel", input, output
            )
            m, n = table["WhereConstantCondConstantScalarYKernel"]
            table["WhereConstantCondConstantScalarYKernel"] = (
                m + default_time_cost,
                n + time_cost,
            )
        else:
            assert False, f"unimplemented: {node.op_type}"
    with open(args.output, "w") as f:
        json.dump(table, f)
