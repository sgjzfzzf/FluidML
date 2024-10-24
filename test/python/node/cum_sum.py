import onnx
import sys

if __name__ == "__main__":
    output_path = sys.argv[1]
    model = onnx.ModelProto(
        ir_version=9,
        opset_import=[onnx.helper.make_opsetid("", 18)],
    )
    graph = model.graph
    graph.name = "cum_sum"
    input = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.INT32,
        [1, 128],
    )
    axis = onnx.helper.make_tensor(
        "axis",
        onnx.TensorProto.INT32,
        [],
        [1],
    )
    output = onnx.helper.make_tensor_value_info(
        "output",
        onnx.TensorProto.INT32,
        [1, 128],
    )
    graph.input.extend([input])
    graph.initializer.extend([axis])
    graph.output.extend([output])
    node = onnx.helper.make_node(
        "CumSum",
        inputs=["input", "axis"],
        outputs=["output"],
        name="cum_sum",
    )
    graph.node.extend([node])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
