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
    graph.name = "gather1"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 384, 136, 1],
    )
    graph.input.extend([input])
    indices = onnx.helper.make_tensor(
        "indices",
        onnx.TensorProto.INT64,
        [9, 128],
        np.random.randint(0, 136, (9, 128)).astype(np.int64),
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 384, 9, 128, 1]
    )
    graph.initializer.extend([indices])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Gather",
        inputs=["input", "indices"],
        outputs=["output"],
        name="gather",
        axis=2,
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
