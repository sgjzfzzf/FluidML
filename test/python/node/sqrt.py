import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "sqrt"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 128, 1],
    )
    output = onnx.helper.make_tensor_value_info(
        "output",
        onnx.TensorProto.FLOAT,
        [1, 128, 1],
    )
    graph.input.extend([input])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Sqrt",
        inputs=["input"],
        outputs=["output"],
        name="sqrt",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
