import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "constant_of_shape"
    input = onnx.helper.make_tensor("input", onnx.TensorProto.INT64, [1], [2])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2])
    graph.initializer.extend([input])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "ConstantOfShape",
        inputs=["input"],
        outputs=["output"],
        name="constant_of_shape",
        value=onnx.helper.make_tensor("value", onnx.TensorProto.INT64, [1], [1]),
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
