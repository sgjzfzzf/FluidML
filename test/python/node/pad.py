import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "pad"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 384, 128, 1],
    )
    pads = onnx.helper.make_tensor(
        "pads",
        onnx.TensorProto.INT64,
        [8],
        [0, 0, 4, 0, 0, 0, 4, 0],
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 384, 136, 1]
    )
    graph.input.extend([input])
    graph.initializer.extend([pads])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Pad",
        inputs=["input", "pads"],
        outputs=["output"],
        name="pad",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
