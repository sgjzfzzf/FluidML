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
    graph.name = "gather_add_add"
    data = onnx.helper.make_tensor(
        "data",
        onnx.TensorProto.FLOAT,
        [30522, 768],
        np.random.rand(30522, 768).astype(np.float32).flatten().tolist(),
    )
    add0_weights = onnx.helper.make_tensor(
        "add0_weights",
        onnx.TensorProto.FLOAT,
        [1, 128, 768],
        np.random.rand(1, 128, 768).astype(np.float32).flatten().tolist(),
    )
    add1_weights = onnx.helper.make_tensor(
        "add1_weights",
        onnx.TensorProto.FLOAT,
        [1, 128, 768],
        np.random.rand(1, 128, 768).astype(np.float32).flatten().tolist(),
    )
    input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.INT64, [1, 128]
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 128, 768]
    )
    graph.input.extend([input])
    graph.initializer.extend([data, add0_weights, add1_weights])
    graph.output.extend([output])
    gather_node = onnx.helper.make_node(
        "Gather",
        inputs=["data", "input"],
        outputs=["gather_output"],
        name="gather",
    )
    add0_node = onnx.helper.make_node(
        "Add",
        inputs=["gather_output", "add0_weights"],
        outputs=["add0_output"],
        name="add0",
    )
    add1_node = onnx.helper.make_node(
        "Add",
        inputs=["add0_output", "add1_weights"],
        outputs=["output"],
        name="add1",
    )
    graph.node.extend([gather_node, add0_node, add1_node])
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
