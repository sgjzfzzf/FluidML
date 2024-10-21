import numpy as np
import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "where0"
    condition = onnx.helper.make_tensor(
        "condition",
        onnx.TensorProto.BOOL,
        [1, 1, 128, 128],
        np.random.randint(0, 2, (1, 1, 128, 128)).astype(np.int32),
    )
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 12, 128, 128],
    )
    other = onnx.helper.make_tensor(
        "other",
        onnx.TensorProto.FLOAT,
        [],
        [-1],
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 12, 128, 128]
    )
    graph.input.extend([input])
    graph.initializer.extend([condition, other])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Where",
        inputs=["condition", "input", "other"],
        outputs=["output"],
        name="where",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
