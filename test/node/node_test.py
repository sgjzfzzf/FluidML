import libCpuTransformers  # type: ignore
import logging
import numpy as np
import onnxruntime
import os
import time
import unittest


class NodeTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(
            filename="node_test.log",
            filemode="w",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

    def test_add0(self):
        add0_onnx_path = os.environ.get("ONNX_add0_PATH")
        self.assertIsNotNone(add0_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(add0_onnx_path, session_options)
        input0 = np.random.random((1, 128, 768)).astype(np.float32)
        input1 = np.random.random((1, 128, 768)).astype(np.float32)
        output = np.zeros((1, 128, 768)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input0": input0,
                "input1": input1,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("add0", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(add0_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input0": input0,
                "input1": input1,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"add0, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_add1(self):
        add1_onnx_path = os.environ.get("ONNX_add1_PATH")
        self.assertIsNotNone(add1_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(add1_onnx_path, session_options)
        input = np.random.random((1, 128, 768)).astype(np.float32)
        output = np.zeros((1, 128, 768)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("add0", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(add1_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"add1, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_div(self):
        div_onnx_path = os.environ.get("ONNX_div_PATH")
        self.assertIsNotNone(div_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(div_onnx_path, session_options)
        input = np.random.random((1, 12, 128, 128)).astype(np.float32)
        output = np.zeros((1, 12, 128, 128)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("div", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(div_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"div, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_erf(self):
        erf_onnx_path = os.environ.get("ONNX_erf_PATH")
        self.assertIsNotNone(erf_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(erf_onnx_path, session_options)
        input = np.random.random((1, 128, 3072)).astype(np.float32)
        output = np.zeros((1, 128, 3072)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("erf", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(erf_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"erf, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_gather_add_add(self):
        gather_add_add_onnx_path = os.environ.get("ONNX_gather_add_add_PATH")
        self.assertIsNotNone(gather_add_add_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(
            gather_add_add_onnx_path, session_options
        )
        input = np.random.randint(0, 30522, (1, 128)).astype(np.int64)
        output = np.zeros((1, 128, 768)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        pm = libCpuTransformers.GraphPassesManager()
        converter = libCpuTransformers.Converter()
        plain_context = libCpuTransformers.Context.Make()
        optim_context = libCpuTransformers.Context.Make()
        plain_builder = libCpuTransformers.NaiveBuilder("gather_add_add", plain_context)
        optim_builder = libCpuTransformers.NaiveBuilder("gather_add_add", optim_context)
        plain_lower = libCpuTransformers.Lower(plain_context)
        optim_lower = libCpuTransformers.Lower(optim_context)
        plain_runner = libCpuTransformers.Runner(plain_context)
        optim_runner = libCpuTransformers.Runner(optim_context)
        plain_graph = parser.Run(gather_add_add_onnx_path)
        optim_graph = parser.Run(gather_add_add_onnx_path)
        pm.RegisterAllPasses()
        pm.Run(optim_graph)
        plain_flow = converter.Run(plain_graph)
        optim_flow = converter.Run(optim_graph)
        planner = libCpuTransformers.GreedyPlanner()
        plain_sequence = planner.FlowToSequence(plain_flow)
        optim_sequence = planner.FlowToSequence(optim_flow)
        plain_index = planner.Run(plain_sequence)
        optim_index = planner.Run(optim_sequence)
        plain_builder.Run(plain_sequence, plain_index)
        optim_builder.Run(optim_sequence, optim_index)
        plain_lower.Run()
        optim_lower.Run()
        plain_timecost = plain_runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        optim_timecost = optim_runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"gather_add_add, onnxruntime timecost: {end - start}, plain timecost: {plain_timecost}, optim timecost: {optim_timecost}"
        )

    def test_gather0(self):
        gather0_onnx_path = os.environ.get("ONNX_gather0_PATH")
        self.assertIsNotNone(gather0_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(gather0_onnx_path, session_options)
        indices = np.random.randint(0, 30522, (1, 128)).astype(np.int64)
        output = np.zeros((1, 128, 768)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "indices": indices,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("gather0", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(gather0_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "indices": indices,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"gather0, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_gather1(self):
        gather1_onnx_path = os.environ.get("ONNX_gather1_PATH")
        self.assertIsNotNone(gather1_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(gather1_onnx_path, session_options)
        data = np.random.random((1, 128, 768)).astype(np.float32)
        output = np.zeros((1, 768)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "data": data,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("gather1", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(gather1_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "data": data,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"gather1, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_gemm(self):
        gemm_onnx_path = os.environ.get("ONNX_gemm_PATH")
        self.assertIsNotNone(gemm_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(gemm_onnx_path, session_options)
        input = np.random.random((1, 768)).astype(np.float32)
        output = np.zeros((1, 768)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("gemm", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(gemm_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"gemm, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_layer_normalization(self):
        layer_normalization_onnx_path = os.environ.get("ONNX_layer_normalization_PATH")
        self.assertIsNotNone(layer_normalization_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(
            layer_normalization_onnx_path, session_options
        )
        input = np.random.random((1, 128, 768)).astype(np.float32)
        output = np.zeros((1, 128, 768)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("layer_normalization", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(layer_normalization_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output, rtol=1e-5))
        self.logger.info(
            f"layer_normalization, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_matmul0(self):
        matmul0_onnx_path = os.environ.get("ONNX_matmul0_PATH")
        self.assertIsNotNone(matmul0_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(matmul0_onnx_path, session_options)
        input0 = np.random.random((1, 12, 128, 64)).astype(np.float32)
        input1 = np.random.random((1, 12, 64, 128)).astype(np.float32)
        output = np.zeros((1, 12, 128, 128)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input0": input0,
                "input1": input1,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("matmul0", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(matmul0_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input0": input0,
                "input1": input1,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output, rtol=1e-5))
        self.logger.info(
            f"matmul0, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_matmul1(self):
        matmul1_onnx_path = os.environ.get("ONNX_matmul1_PATH")
        self.assertIsNotNone(matmul1_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(matmul1_onnx_path, session_options)
        input = np.random.random((1, 128, 768)).astype(np.float32)
        output = np.zeros((1, 128, 768)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("matmul1", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(matmul1_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"matmul1, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_mul0(self):
        mul0_onnx_path = os.environ.get("ONNX_mul0_PATH")
        self.assertIsNotNone(mul0_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(mul0_onnx_path, session_options)
        input0 = np.random.random((1, 128, 3072)).astype(np.float32)
        input1 = np.random.random((1, 128, 3072)).astype(np.float32)
        output = np.zeros((1, 128, 3072)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input0": input0,
                "input1": input1,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("mul0", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(mul0_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input0": input0,
                "input1": input1,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"mul0, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_mul1(self):
        mul1_onnx_path = os.environ.get("ONNX_mul1_PATH")
        self.assertIsNotNone(mul1_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(mul1_onnx_path, session_options)
        input = np.random.random((1, 1, 1, 128)).astype(np.float32)
        output = np.zeros((1, 1, 1, 128)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("mul1", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(mul1_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"mul1, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_pow(self):
        pow_onnx_path = os.environ.get("ONNX_pow_PATH")
        self.assertIsNotNone(pow_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(pow_onnx_path, session_options)
        input = np.random.random((1, 128, 3072)).astype(np.float32)
        output = np.zeros((1, 128, 3072)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("pow", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(pow_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"pow, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_reshape(self):
        reshape_onnx_path = os.environ.get("ONNX_reshape_PATH")
        self.assertIsNotNone(reshape_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(reshape_onnx_path, session_options)
        data = np.random.random((1, 128, 768)).astype(np.float32)
        output = np.zeros((1, 128, 12, 64)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "data": data,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("reshape", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(reshape_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "data": data,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"reshape, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_softmax(self):
        softmax_onnx_path = os.environ.get("ONNX_softmax_PATH")
        self.assertIsNotNone(softmax_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(softmax_onnx_path, session_options)
        input = np.random.random((1, 12, 128, 768)).astype(np.float32)
        output = np.zeros((1, 12, 128, 768)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("softmax", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(softmax_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"softmax, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    # TODO: bugs in the runner.Run, remember to check the reason
    def test_split(self):
        split_onnx_path = os.environ.get("ONNX_split_PATH")
        self.assertIsNotNone(split_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(split_onnx_path, session_options)
        input = np.random.random((1, 128, 2304)).astype(np.float32)
        output0 = np.zeros((1, 128, 768)).astype(np.float32)
        output1 = np.zeros((1, 128, 768)).astype(np.float32)
        output2 = np.zeros((1, 128, 768)).astype(np.float32)
        (onnx_output0, onnx_output1, onnx_output2) = session.run(
            ["output0", "output1", "output2"],
            {
                "input": input,
            },
        )
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("split", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(split_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        # runner.Run(
        #     {
        #         "input": input,
        #         "output0": output0,
        #         "output1": output1,
        #         "output2": output2,
        #     }
        # )
        # self.assertTrue(np.allclose(onnx_output0, output0))
        # self.assertTrue(np.allclose(onnx_output1, output1))
        # self.assertTrue(np.allclose(onnx_output2, output2))

    def test_sub(self):
        sub_onnx_path = os.environ.get("ONNX_sub_PATH")
        self.assertIsNotNone(sub_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(sub_onnx_path, session_options)
        input0 = np.random.random((1, 1, 1, 128)).astype(np.float32)
        output = np.zeros((1, 1, 1, 128)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input0,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("sub", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(sub_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input0,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"sub, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_tanh(self):
        tanh_onnx_path = os.environ.get("ONNX_tanh_PATH")
        self.assertIsNotNone(tanh_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(tanh_onnx_path, session_options)
        input = np.random.random((1, 768)).astype(np.float32)
        output = np.zeros((1, 768)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("tanh", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(tanh_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"tanh, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_transpose(self):
        transpose_onnx_path = os.environ.get("ONNX_transpose_PATH")
        self.assertIsNotNone(transpose_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(transpose_onnx_path, session_options)
        input = np.random.random((1, 128, 12, 64)).astype(np.float32)
        output = np.zeros((1, 12, 128, 64)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("transpose", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(transpose_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"transpose, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_unsqueeze(self):
        unsqueeze_onnx_path = os.environ.get("ONNX_unsqueeze_PATH")
        self.assertIsNotNone(unsqueeze_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(unsqueeze_onnx_path, session_options)
        input = np.random.random((1, 128)).astype(np.float32)
        output = np.zeros((1, 1, 1, 128)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("unsqueeze", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(unsqueeze_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"unsqueeze, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_where0(self):
        where_onnx_path = os.environ.get("ONNX_where_PATH")
        self.assertIsNotNone(where_onnx_path)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(where_onnx_path, session_options)
        input = np.random.random((1, 12, 128, 128)).astype(np.float32)
        output = np.zeros((1, 12, 128, 128)).astype(np.float32)
        start = time.time_ns()
        (onnx_output,) = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        end = time.time_ns()
        parser = libCpuTransformers.Parser()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("where", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        graph = parser.Run(where_onnx_path)
        flow = converter.Run(graph)
        planner = libCpuTransformers.GreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        timecost = runner.Run(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"where, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )


if __name__ == "__main__":
    unittest.main()
