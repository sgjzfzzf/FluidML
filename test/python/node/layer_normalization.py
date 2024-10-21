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
    graph.name = "layer_normalization"
    input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, 128, 768]
    )
    graph.input.extend([input])
    scale = onnx.helper.make_tensor(
        "scale", onnx.TensorProto.FLOAT, [768], np.ones((768)).astype(np.float32)
    )
    bias = onnx.helper.make_tensor(
        "bias", onnx.TensorProto.FLOAT, [768], np.zeros((768)).astype(np.float32)
    )
    graph.initializer.extend([scale, bias])
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 128, 768]
    )
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "LayerNormalization",
        inputs=["input", "scale", "bias"],
        outputs=["output"],
        name="layer_normalization",
        axis=-1,
        epsilon=1e-12,
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
