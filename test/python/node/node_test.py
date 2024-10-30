import cpu_transformers  # type: ignore
import logging
import numpy as np
import onnxruntime
import os
import time
import unittest

from typing import Tuple


class NodeTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(
            filename="node_test.log",
            filemode="w",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

    def test_add0(self):
        name: str = "add0"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        lhs: np.ndarray = np.random.random((1, 128, 768)).astype(np.float32)
        rhs: np.ndarray = np.random.random((1, 128, 768)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 128, 768)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "lhs": lhs,
                "rhs": rhs,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "lhs": lhs,
                "rhs": rhs,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_add1(self):
        name: str = "add1"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 128, 768)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 128, 768)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_cast(self):
        name: str = "cast"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.randint(0, 2, (1, 128)).astype(np.bool)
        output: np.ndarray = np.zeros((1, 128)).astype(np.int32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_concat(self):
        name: str = "concat"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        lhs: np.ndarray = np.random.random((1, 4, 128, 2)).astype(np.float32)
        rhs: np.ndarray = np.random.random((1, 4, 128, 6)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 4, 128, 8)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "lhs": lhs,
                "rhs": rhs,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "lhs": lhs,
                "rhs": rhs,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_conv0(self):
        name: str = "conv0"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 768, 128)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 768, 128)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_conv1(self):
        name: str = "conv1"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 768, 128)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 384, 128)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output, rtol=1e-2))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_cum_sum(self):
        name: str = "cum_sum"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.randint(0, 2, (1, 128)).astype(np.int32)
        output: np.ndarray = np.zeros((1, 128)).astype(np.int32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_div0(self):
        name: str = "div0"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        lhs: np.ndarray = np.random.random((1, 128, 768)).astype(np.float32)
        rhs: np.ndarray = np.random.random((1, 128, 1)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 128, 768)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "lhs": lhs,
                "rhs": rhs,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "lhs": lhs,
                "rhs": rhs,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_div1(self):
        name: str = "div1"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 12, 128, 128)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 12, 128, 128)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_equal(self):
        name: str = "equal"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.randint(0, 2, (1, 128)).astype(np.int64)
        output: np.ndarray = np.zeros((1, 128)).astype(np.bool)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_erf(self):
        name: str = "erf"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 128, 3072)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 128, 3072)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_gather_add_add(self):
        name: str = "gather_add_add"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.randint(0, 30522, (1, 128)).astype(np.int64)
        output: np.ndarray = np.zeros((1, 128, 768)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_gather0(self):
        name: str = "gather0"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.randint(0, 30522, (1, 128)).astype(np.int64)
        output: np.ndarray = np.zeros((1, 128, 768)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_gather1(self):
        name: str = "gather1"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 384, 136, 1)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 384, 9, 128, 1)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_gather2(self):
        name: str = "gather2"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 128, 768)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 768)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_gemm(self):
        name: str = "gemm"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 768)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 768)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_layer_normalization(self):
        name: str = "layer_normalization"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 128, 768)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 128, 768)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_matmul0(self):
        name: str = "matmul0"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        lhs: np.ndarray = np.random.random((1, 12, 128, 64)).astype(np.float32)
        rhs: np.ndarray = np.random.random((1, 12, 64, 128)).astype(np.float32)
        output = np.zeros((1, 12, 128, 128)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "lhs": lhs,
                "rhs": rhs,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "lhs": lhs,
                "rhs": rhs,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_matmul1(self):
        name: str = "matmul1"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 128, 768)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 128, 768)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_mul0(self):
        name: str = "mul0"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        lhs: np.ndarray = np.random.random((1, 128, 3072)).astype(np.float32)
        rhs: np.ndarray = np.random.random((1, 128, 3072)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 128, 3072)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "lhs": lhs,
                "rhs": rhs,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "lhs": lhs,
                "rhs": rhs,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_mul1(self):
        name: str = "mul1"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 1, 1, 128)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 1, 1, 128)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_neg(self):
        name: str = "neg"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 4, 128, 1)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 4, 128, 1)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_not(self):
        name: str = "not"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.randint(0, 2, (1, 128)).astype(np.bool)
        output: np.ndarray = np.zeros((1, 128)).astype(np.bool)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_pad(self):
        name: str = "pad"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 384, 128, 1)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 384, 136, 1)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_pow(self):
        name: str = "pow"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 128, 3072)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 128, 3072)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_reduce_mean(self):
        name: str = "reduce_mean"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 128, 768)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 128, 1)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_reshape(self):
        name: str = "reshape"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 128, 768)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 128, 12, 64)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"reshape, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_slice(self):
        name: str = "slice"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 128, 4, 24)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 128, 4, 8)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_softmax(self):
        name: str = "softmax"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 12, 128, 768)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 12, 128, 768)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_sqrt(self):
        name: str = "sqrt"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 128, 1)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 128, 1)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"sqrt, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_sub0(self):
        name: str = "sub0"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        lhs: np.ndarray = np.random.random((1, 128, 768)).astype(np.float32)
        rhs: np.ndarray = np.random.random((1, 128, 1)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 128, 768)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "lhs": lhs,
                "rhs": rhs,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "lhs": lhs,
                "rhs": rhs,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_sub1(self):
        name: str = "sub1"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 1, 1, 128)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 1, 1, 128)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_tanh(self):
        name: str = "tanh"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 768)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 768)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_transpose(self):
        name: str = "transpose"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 128, 12, 64)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 12, 128, 64)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_unsqueeze(self):
        name: str = "unsqueeze"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 128)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 1, 1, 128)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )

    def test_where(self):
        name: str = "where"
        onnx_path: str = os.environ.get(f"ONNX_{name}_PATH")
        self.assertIsNotNone(onnx_path)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            onnx_path, session_options
        )
        input: np.ndarray = np.random.random((1, 12, 128, 128)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 12, 128, 128)).astype(np.float32)
        start: int = time.time_ns()
        onnx_output_tuple: Tuple[np.ndarray] = session.run(
            ["output"],
            {
                "input": input,
            },
        )
        (onnx_output,) = onnx_output_tuple
        end: int = time.time_ns()
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
        executor.compile(onnx_path)
        timecost: int = executor.invoke(
            {
                "input": input,
                "output": output,
            }
        )
        self.assertTrue(np.allclose(onnx_output, output))
        self.logger.info(
            f"{name}, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )


if __name__ == "__main__":
    unittest.main()
