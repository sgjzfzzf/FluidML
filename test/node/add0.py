import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "add0"
    input0 = onnx.helper.make_tensor_value_info(
        "input0", onnx.TensorProto.FLOAT, [1, 128, 768]
    )
    input1 = onnx.helper.make_tensor_value_info(
        "input1", onnx.TensorProto.FLOAT, [1, 128, 768]
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 128, 768]
    )
    graph.input.extend([input0, input1])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Add",
        inputs=["input0", "input1"],
        outputs=["output"],
        name="add",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
