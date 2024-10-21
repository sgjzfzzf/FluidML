import cpu_transformers  # type: ignore
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
        executor: cpu_transformers.Executor = (
            cpu_transformers.Executor.make_plain_greedy(name)
        )
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
            f"Time cost: {time_cost} ns\nONNX time cost: {onnx_time_cost} ns"
        )


if __name__ == "__main__":
    unittest.main()
