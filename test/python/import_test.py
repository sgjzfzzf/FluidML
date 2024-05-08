import libCpuTransformers  # type: ignore
import unittest


class ImportTest(unittest.TestCase):
    def test_context(self):
        self.assertTrue(hasattr(libCpuTransformers, "Context"))

    def test_graph(self):
        self.assertTrue(hasattr(libCpuTransformers, "Graph"))

    def test_flow(self):
        self.assertTrue(hasattr(libCpuTransformers, "Flow"))

    def test_sequence(self):
        self.assertTrue(hasattr(libCpuTransformers, "Sequence"))

    def test_index(self):
        self.assertTrue(hasattr(libCpuTransformers, "Index"))

    def test_plan(self):
        self.assertTrue(hasattr(libCpuTransformers, "Plan"))

    def test_naive_builder(self):
        self.assertTrue(hasattr(libCpuTransformers, "NaiveBuilder"))

    def test_converter(self):
        self.assertTrue(hasattr(libCpuTransformers, "Converter"))

    def test_lower(self):
        self.assertTrue(hasattr(libCpuTransformers, "Lower"))

    def test_parser(self):
        self.assertTrue(hasattr(libCpuTransformers, "Parser"))

    def test_linear_planner(self):
        self.assertTrue(hasattr(libCpuTransformers, "LinearPlanner"))

    def test_runner(self):
        self.assertTrue(hasattr(libCpuTransformers, "Runner"))


if __name__ == "__main__":
    unittest.main()
