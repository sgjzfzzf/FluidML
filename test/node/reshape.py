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
    graph.name = "reshape"
    data = onnx.helper.make_tensor_value_info(
        "data",
        onnx.TensorProto.FLOAT,
        [1, 128, 768],
    )
    shape = onnx.helper.make_tensor(
        "shape",
        onnx.TensorProto.INT64,
        [4],
        np.array([1, 128, 12, 64], dtype=np.int64),
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 128, 12, 64]
    )
    graph.input.extend([data])
    graph.initializer.extend([shape])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Reshape",
        inputs=["data", "shape"],
        outputs=["output"],
        name="reshape",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
