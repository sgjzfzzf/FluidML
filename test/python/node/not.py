import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "not"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.BOOL,
        [1, 128],
    )
    output = onnx.helper.make_tensor_value_info(
        "output",
        onnx.TensorProto.BOOL,
        [1, 128],
    )
    graph.input.extend([input])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Not",
        inputs=["input"],
        outputs=["output"],
        name="not",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
