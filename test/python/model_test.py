import libCpuTransformers  # type: ignore
import logging
import numpy as np
import onnxruntime
import os
import time
import unittest


class ModelTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(
            filename="model_test.log",
            filemode="w",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

    def test_bert(self):
        parser = libCpuTransformers.Parser()
        pm = libCpuTransformers.GraphPassesManager()
        converter = libCpuTransformers.Converter()
        context = libCpuTransformers.Context.Make()
        builder = libCpuTransformers.NaiveBuilder("bert", context)
        lower = libCpuTransformers.Lower(context)
        runner = libCpuTransformers.Runner(context)
        model_path = os.environ.get("BERT_MODEL_PATH")
        self.assertIsNotNone(model_path)
        graph = parser.Run(model_path)
        pm.RegisterAllPasses()
        pm.Run(graph)
        flow = converter.Run(graph)
        planner = libCpuTransformers.PlainGreedyPlanner()
        sequence = planner.FlowToSequence(flow)
        index = planner.Run(sequence)
        builder.Run(sequence, index)
        lower.Run()
        input_ids = np.random.randint(0, 30522, (1, 128)).astype(np.int64)
        attention_mask = np.random.randint(0, 2, (1, 128)).astype(np.float32)
        output0 = np.zeros((1, 128, 768), dtype=np.float32)
        output1 = np.zeros((1, 768), dtype=np.float32)
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(model_path, session_options)
        start = time.time_ns()
        session.run(
            ["onnx::Gather_1269", "1272"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        end = time.time_ns()
        timecost = runner.Run(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "onnx::Gather_1269": output0,
                "1272": output1,
            }
        )
        self.logger.info(
            f"bert, onnxruntime timecost: {end - start}, timecost: {timecost}"
        )


if __name__ == "__main__":
    unittest.main()
