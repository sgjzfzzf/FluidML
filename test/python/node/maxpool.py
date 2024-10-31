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
    graph.name = "maxpool"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 64, 224, 224],
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 64, 112, 112]
    )
    graph.input.extend([input])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "MaxPool",
        inputs=["input"],
        outputs=["output"],
        name="maxpool",
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        strides=[2, 2],
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
