import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "split"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 128, 2304],
    )
    sizes = onnx.helper.make_tensor(
        "sizes",
        onnx.TensorProto.INT64,
        [3],
        [768, 768, 768],
    )
    output0 = onnx.helper.make_tensor_value_info(
        "output0", onnx.TensorProto.FLOAT, [1, 128, 768]
    )
    output1 = onnx.helper.make_tensor_value_info(
        "output1", onnx.TensorProto.FLOAT, [1, 128, 768]
    )
    output2 = onnx.helper.make_tensor_value_info(
        "output2", onnx.TensorProto.FLOAT, [1, 128, 768]
    )
    graph.input.extend([input])
    graph.initializer.extend([sizes])
    graph.output.extend([output0, output1, output2])
    node = onnx.helper.make_node(
        "Split",
        inputs=["input", "sizes"],
        outputs=["output0", "output1", "output2"],
        name="split",
        axis=2,
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
