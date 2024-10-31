import os

print(os.environ.get("PYTHONPATH"))


import fluidml  # type: ignore
import unittest


class ImportTest(unittest.TestCase):
    def test_context(self):
        self.assertTrue(hasattr(fluidml, "Context"))

    def test_flow(self):
        self.assertTrue(hasattr(fluidml, "Flow"))

    def test_sequence(self):
        self.assertTrue(hasattr(fluidml, "Sequence"))

    def test_graph(self):
        self.assertTrue(hasattr(fluidml, "Graph"))

    def test_index(self):
        self.assertTrue(hasattr(fluidml, "Index"))

    def test_builder(self):
        self.assertTrue(hasattr(fluidml, "Builder"))

    def test_general_builder(self):
        self.assertTrue(hasattr(fluidml, "GeneralBuilder"))
        self.assertTrue(hasattr(fluidml.GeneralBuilder, "run"))

    def test_converter(self):
        self.assertTrue(hasattr(fluidml, "Converter"))
        self.assertTrue(hasattr(fluidml.Converter, "run"))
        self.assertTrue(hasattr(fluidml.Converter, "make"))

    def test_executor(self):
        self.assertTrue(hasattr(fluidml, "Executor"))
        self.assertTrue(hasattr(fluidml.Executor, "compile"))
        self.assertTrue(hasattr(fluidml.Executor, "invoke"))
        self.assertTrue(hasattr(fluidml.Executor, "make_plain_linear"))
        self.assertTrue(hasattr(fluidml.Executor, "make_plain_greedy"))
        self.assertTrue(hasattr(fluidml.Executor, "make_dp_greedy"))

    def test_lower(self):
        self.assertTrue(hasattr(fluidml, "Lower"))
        self.assertTrue(hasattr(fluidml.Lower, "run"))

    def test_parser(self):
        self.assertTrue(hasattr(fluidml, "Parser"))
        self.assertTrue(hasattr(fluidml.Parser, "run"))
        self.assertTrue(hasattr(fluidml.Parser, "make"))

    def test_planner(self):
        self.assertTrue(hasattr(fluidml, "Planner"))
        self.assertTrue(hasattr(fluidml.Planner, "run"))

    def test_runner(self):
        self.assertTrue(hasattr(fluidml, "Runner"))
        self.assertTrue(hasattr(fluidml.Runner, "run"))


if __name__ == "__main__":
    unittest.main()
