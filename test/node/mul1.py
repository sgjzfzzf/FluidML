import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "mul1"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 1, 1, 128],
    )
    weights = onnx.helper.make_tensor("weights", onnx.TensorProto.FLOAT, [], [1])
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 1, 1, 128]
    )
    graph.input.extend([input])
    graph.initializer.extend([weights])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Mul",
        inputs=["input", "weights"],
        outputs=["output"],
        name="mul",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
