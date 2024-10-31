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
    graph.name = "conv2"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 512, 14, 14],
    )
    weights = onnx.helper.make_tensor(
        "weights",
        onnx.TensorProto.FLOAT,
        [512, 512, 3, 3],
        np.random.rand(512, 512, 3, 3).astype(np.float32),
    )
    bias = onnx.helper.make_tensor(
        "bias",
        onnx.TensorProto.FLOAT,
        [512],
        np.random.rand(512).astype(np.float32),
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 512, 14, 14]
    )
    graph.input.extend([input])
    graph.initializer.extend([weights, bias])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Conv",
        inputs=["input", "weights", "bias"],
        outputs=["output"],
        name="conv",
        dilations=[1, 1],
        group=1,
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
