import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "matmul0"
    lhs = onnx.helper.make_tensor_value_info(
        "lhs", onnx.TensorProto.FLOAT, [1, 12, 128, 64]
    )
    rhs = onnx.helper.make_tensor_value_info(
        "rhs", onnx.TensorProto.FLOAT, [1, 12, 64, 128]
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 12, 128, 128]
    )
    graph.input.extend([lhs, rhs])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "MatMul",
        inputs=["lhs", "rhs"],
        outputs=["output"],
        name="matmul",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
