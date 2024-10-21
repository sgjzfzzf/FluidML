import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "unsqueeze"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 128],
    )
    axes = onnx.helper.make_tensor(
        "axes",
        onnx.TensorProto.INT64,
        [2],
        [1, 2],
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 1, 1, 128]
    )
    graph.input.extend([input])
    graph.initializer.extend([axes])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Unsqueeze",
        inputs=["input", "axes"],
        outputs=["output"],
        name="unsqueeze",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
