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
    graph.name = "conv0"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 768, 128],
    )
    weights = onnx.helper.make_tensor(
        "weights",
        onnx.TensorProto.FLOAT,
        [768, 1, 9],
        np.random.rand(768, 1, 9).astype(np.float32),
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 768, 128]
    )
    graph.input.extend([input])
    graph.initializer.extend([weights])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "Conv",
        inputs=["input", "weights"],
        outputs=["output"],
        name="conv",
        dilations=[1],
        group=768,
        kernel_shape=[9],
        pads=[4, 4],
        strides=[1],
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
