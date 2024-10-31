import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "dropout"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 4096],
    )
    ratio = onnx.helper.make_tensor(
        name="ratio",
        data_type=onnx.TensorProto.FLOAT,
        dims=[],
        vals=[0.5],
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 4096]
    )
    graph.input.extend([input])
    graph.initializer.extend([ratio])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Dropout",
        inputs=["input", "ratio"],
        outputs=["output"],
        name="dropout",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
