import fluidml  # type: ignore
import logging
import numpy as np
import onnxruntime
import os
import time
import unittest

from typing import Optional


class ModelTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(
            filename="model_test.log",
            filemode="w",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

    def test_bert(self):
        name: str = "bert"
        input: Optional[str] = os.environ.get("BERT_MODEL_PATH")
        self.assertIsNotNone(input)
        mlir: str = f"{name}.mlir"
        llvm: str = f"{name}-llvm.mlir"
        executor: fluidml.Executor = fluidml.Executor.make_plain_greedy(name)
        executor.compile(input, mlir, llvm)
        input_ids: np.ndarray = np.random.randint(0, 30522, (1, 128), dtype=np.int64)
        attention_mask: np.ndarray = np.ones((1, 128), dtype=np.float32)
        output0: np.ndarray = np.zeros((1, 128, 768), dtype=np.float32)
        output1: np.ndarray = np.zeros((1, 768), dtype=np.float32)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            input, session_options
        )
        start: int = time.time_ns()
        session.run(
            [
                "onnx::Gather_1269",
                "1272",
            ],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        end: int = time.time_ns()
        onnx_time_cost: int = end - start
        time_cost: int = executor.invoke(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "onnx::Gather_1269": output0,
                "1272": output1,
            }
        )
        self.logger.info(
            f"{name}:\nTime cost: {time_cost} ns\nONNX time cost: {onnx_time_cost} ns"
        )

    def test_convbert(self):
        name: str = "convbert"
        input: Optional[str] = os.environ.get("CONVBERT_MODEL_PATH")
        self.assertIsNotNone(input)
        mlir: str = f"{name}.mlir"
        llvm: str = f"{name}-llvm.mlir"
        executor: fluidml.Executor = fluidml.Executor.make_plain_greedy(name)
        executor.compile(input, mlir, llvm)
        input_ids: np.ndarray = np.random.randint(0, 30522, (1, 128), dtype=np.int64)
        attention_mask: np.ndarray = np.ones((1, 128), dtype=np.float32)
        output: np.ndarray = np.zeros((1, 128, 768), dtype=np.float32)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            input, session_options
        )
        start: int = time.time_ns()
        session.run(
            [
                "2625",
            ],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        end: int = time.time_ns()
        onnx_time_cost: int = end - start
        time_cost: int = executor.invoke(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "2625": output,
            }
        )
        self.logger.info(
            f"{name}:\nTime cost: {time_cost} ns\nONNX time cost: {onnx_time_cost} ns"
        )

    def test_efficientnet(self):
        name: str = "efficientnet"
        input: Optional[str] = os.environ.get("EFFICIENTNET_MODEL_PATH")
        self.assertIsNotNone(input)
        mlir: str = f"{name}.mlir"
        llvm: str = f"{name}-llvm.mlir"
        executor: fluidml.Executor = fluidml.Executor.make_plain_greedy(name)
        executor.compile(input, mlir, llvm)
        data: np.ndarray = np.random.random((1, 224, 224, 3)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 1000), dtype=np.float32)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            input, session_options
        )
        start: int = time.time_ns()
        session.run(
            [
                "Softmax:0",
            ],
            {
                "images:0": data,
            },
        )
        end: int = time.time_ns()
        onnx_time_cost: int = end - start
        time_cost: int = executor.invoke(
            {
                "images:0": data,
                "Softmax:0": output,
            }
        )
        self.logger.info(
            f"{name}:\nTime cost: {time_cost} ns\nONNX time cost: {onnx_time_cost} ns"
        )

    def test_gptneox(self):
        name: str = "gptneox"
        input: Optional[str] = os.environ.get("GPTNEOX_MODEL_PATH")
        self.assertIsNotNone(input)
        mlir: str = f"{name}.mlir"
        llvm: str = f"{name}-llvm.mlir"
        executor: fluidml.Executor = fluidml.Executor.make_plain_greedy(name)
        executor.compile(input, mlir, llvm)
        input_ids: np.ndarray = np.random.randint(0, 1024, (1, 128), dtype=np.int64)
        attention_mask: np.ndarray = np.ones((1, 128), dtype=np.float32)
        output0: np.ndarray = np.zeros((1, 128, 32), dtype=np.float32)
        output1: np.ndarray = np.zeros((1, 4, 128, 8), dtype=np.float32)
        output2: np.ndarray = np.zeros((1, 4, 128, 8), dtype=np.float32)
        output3: np.ndarray = np.zeros((1, 4, 128, 8), dtype=np.float32)
        output4: np.ndarray = np.zeros((1, 4, 128, 8), dtype=np.float32)
        output5: np.ndarray = np.zeros((1, 4, 128, 8), dtype=np.float32)
        output6: np.ndarray = np.zeros((1, 4, 128, 8), dtype=np.float32)
        output7: np.ndarray = np.zeros((1, 4, 128, 8), dtype=np.float32)
        output8: np.ndarray = np.zeros((1, 4, 128, 8), dtype=np.float32)
        output9: np.ndarray = np.zeros((1, 4, 128, 8), dtype=np.float32)
        output10: np.ndarray = np.zeros((1, 4, 128, 8), dtype=np.float32)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            input, session_options
        )
        start: int = time.time_ns()
        session.run(
            [
                "1182",
                "key",
                "value",
                "key.3",
                "value.3",
                "key.7",
                "value.7",
                "key.11",
                "value.11",
                "key.15",
                "value.15",
            ],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        end: int = time.time_ns()
        onnx_time_cost: int = end - start
        time_cost: int = executor.invoke(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "1182": output0,
                "key": output1,
                "value": output2,
                "key.3": output3,
                "value.3": output4,
                "key.7": output5,
                "value.7": output6,
                "key.11": output7,
                "value.11": output8,
                "key.15": output9,
                "value.15": output10,
            }
        )
        self.logger.info(
            f"{name}:\nTime cost: {time_cost} ns\nONNX time cost: {onnx_time_cost} ns"
        )

    def test_ibert(self):
        name: str = "ibert"
        input: Optional[str] = os.environ.get("IBERT_MODEL_PATH")
        self.assertIsNotNone(input)
        mlir: str = f"{name}.mlir"
        llvm: str = f"{name}-llvm.mlir"
        executor: fluidml.Executor = fluidml.Executor.make_plain_greedy(name)
        executor.compile(input, mlir, llvm)
        input_ids: np.ndarray = np.random.randint(0, 50265, (1, 128), dtype=np.int64)
        attention_mask: np.ndarray = np.ones((1, 128), dtype=np.float32)
        output0: np.ndarray = np.zeros((1, 128, 768), dtype=np.float32)
        output1: np.ndarray = np.zeros((1, 768), dtype=np.float32)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            input, session_options
        )
        start: int = time.time_ns()
        session.run(
            [
                "onnx::Gather_2276",
                "2279",
            ],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        end: int = time.time_ns()
        onnx_time_cost: int = end - start
        time_cost: int = executor.invoke(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "onnx::Gather_2276": output0,
                "2279": output1,
            }
        )
        self.logger.info(
            f"{name}:\nTime cost: {time_cost} ns\nONNX time cost: {onnx_time_cost} ns"
        )

    def test_vgg(self):
        name: str = "vgg"
        input: Optional[str] = os.environ.get("VGG_MODEL_PATH")
        self.assertIsNotNone(input)
        mlir: str = f"{name}.mlir"
        llvm: str = f"{name}-llvm.mlir"
        executor: fluidml.Executor = fluidml.Executor.make_plain_greedy(name)
        executor.compile(input, mlir, llvm)
        data: np.ndarray = np.random.random((1, 3, 224, 224)).astype(np.float32)
        output: np.ndarray = np.zeros((1, 1000), dtype=np.float32)
        session_options: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            input, session_options
        )
        start: int = time.time_ns()
        session.run(
            [
                "vgg0_dense2_fwd",
            ],
            {
                "data": data,
            },
        )
        end: int = time.time_ns()
        onnx_time_cost: int = end - start
        time_cost: int = executor.invoke(
            {
                "data": data,
                "vgg0_dense2_fwd": output,
            }
        )
        self.logger.info(
            f"{name}:\nTime cost: {time_cost} ns\nONNX time cost: {onnx_time_cost} ns"
        )


if __name__ == "__main__":
    unittest.main()
