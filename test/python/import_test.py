import os

print(os.environ.get("PYTHONPATH"))


import cpu_transformers  # type: ignore
import unittest


class ImportTest(unittest.TestCase):
    def test_context(self):
        self.assertTrue(hasattr(cpu_transformers, "Context"))

    def test_flow(self):
        self.assertTrue(hasattr(cpu_transformers, "Flow"))

    def test_sequence(self):
        self.assertTrue(hasattr(cpu_transformers, "Sequence"))

    def test_graph(self):
        self.assertTrue(hasattr(cpu_transformers, "Graph"))

    def test_index(self):
        self.assertTrue(hasattr(cpu_transformers, "Index"))

    def test_builder(self):
        self.assertTrue(hasattr(cpu_transformers, "Builder"))

    def test_general_builder(self):
        self.assertTrue(hasattr(cpu_transformers, "GeneralBuilder"))
        self.assertTrue(hasattr(cpu_transformers.GeneralBuilder, "run"))

    def test_converter(self):
        self.assertTrue(hasattr(cpu_transformers, "Converter"))
        self.assertTrue(hasattr(cpu_transformers.Converter, "run"))
        self.assertTrue(hasattr(cpu_transformers.Converter, "make"))

    def test_executor(self):
        self.assertTrue(hasattr(cpu_transformers, "Executor"))
        self.assertTrue(hasattr(cpu_transformers.Executor, "compile"))
        self.assertTrue(hasattr(cpu_transformers.Executor, "invoke"))
        self.assertTrue(hasattr(cpu_transformers.Executor, "make_plain_linear"))
        self.assertTrue(hasattr(cpu_transformers.Executor, "make_plain_greedy"))
        self.assertTrue(hasattr(cpu_transformers.Executor, "make_dp_greedy"))

    def test_lower(self):
        self.assertTrue(hasattr(cpu_transformers, "Lower"))
        self.assertTrue(hasattr(cpu_transformers.Lower, "run"))

    def test_parser(self):
        self.assertTrue(hasattr(cpu_transformers, "Parser"))
        self.assertTrue(hasattr(cpu_transformers.Parser, "run"))
        self.assertTrue(hasattr(cpu_transformers.Parser, "make"))

    def test_planner(self):
        self.assertTrue(hasattr(cpu_transformers, "Planner"))
        self.assertTrue(hasattr(cpu_transformers.Planner, "run"))

    def test_runner(self):
        self.assertTrue(hasattr(cpu_transformers, "Runner"))
        self.assertTrue(hasattr(cpu_transformers.Runner, "run"))


if __name__ == "__main__":
    unittest.main()
