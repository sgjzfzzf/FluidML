import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "slice"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 128, 4, 24],
    )
    starts = onnx.helper.make_tensor(
        "starts",
        onnx.TensorProto.INT64,
        [1],
        [8],
    )
    ends = onnx.helper.make_tensor(
        "ends",
        onnx.TensorProto.INT64,
        [1],
        [16],
    )
    axes = onnx.helper.make_tensor(
        "axes",
        onnx.TensorProto.INT64,
        [1],
        [3],
    )
    steps = onnx.helper.make_tensor(
        "steps",
        onnx.TensorProto.INT64,
        [1],
        [1],
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 128, 4, 8]
    )
    graph.input.extend([input])
    graph.initializer.extend([starts, ends, axes, steps])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Slice",
        inputs=["input", "starts", "ends", "axes", "steps"],
        outputs=["output"],
        name="slice",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
