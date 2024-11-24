import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "clip"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 32, 112, 112],
    )
    max = onnx.helper.make_tensor(
        "max",
        onnx.TensorProto.FLOAT,
        [],
        [6],
    )
    min = onnx.helper.make_tensor(
        "min",
        onnx.TensorProto.FLOAT,
        [],
        [0],
    )
    output = onnx.helper.make_tensor_value_info(
        "output",
        onnx.TensorProto.FLOAT,
        [1, 32, 112, 112],
    )
    graph.input.extend([input])
    graph.initializer.extend([max, min])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Clip",
        inputs=["input", "max", "min"],
        outputs=["output"],
        name="clip",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
