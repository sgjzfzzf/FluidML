import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "equal"
    input0 = onnx.helper.make_tensor("input0", onnx.TensorProto.INT64, [2], [1, 128])
    input1 = onnx.helper.make_tensor_value_info(
        "input1",
        onnx.TensorProto.INT64,
        [2],
    )
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, [2])
    graph.initializer.extend([input0])
    graph.input.extend([input1])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Equal",
        inputs=["input0", "input1"],
        outputs=["output"],
        name="equal",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
