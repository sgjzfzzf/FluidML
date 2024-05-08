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
    graph.name = "gather0"
    data = onnx.helper.make_tensor(
        "data",
        onnx.TensorProto.FLOAT,
        [30522, 768],
        np.random.rand(30522, 768).astype(np.float32).flatten().tolist(),
    )
    graph.initializer.extend([data])
    indices = onnx.helper.make_tensor_value_info(
        "indices", onnx.TensorProto.INT64, [1, 128]
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 128, 768]
    )
    graph.input.extend([indices])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Gather",
        inputs=["data", "indices"],
        outputs=["output"],
        name="gather",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
