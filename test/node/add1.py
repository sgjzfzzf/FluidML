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
    graph.name = "add1"
    input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, 128, 768]
    )
    weights = onnx.helper.make_tensor(
        "weights",
        onnx.TensorProto.FLOAT,
        [1, 128, 768],
        np.random.rand(1, 128, 768).astype(np.float32).flatten().tolist(),
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 128, 768]
    )
    graph.input.extend([input])
    graph.initializer.extend([weights])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Add",
        inputs=["input", "weights"],
        outputs=["output"],
        name="add",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
