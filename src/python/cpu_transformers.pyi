import numpy as np
from typing import Dict, Optional, Tuple

class Context:
    @classmethod
    def __init__() -> None: ...
    @classmethod
    def make_plain_general_builder(self, function_name: str) -> GeneralBuilder: ...
    @classmethod
    def make_dp_general_builder(self, function_name: str) -> GeneralBuilder: ...
    @classmethod
    def make_kernel_builder(self, function_name: str) -> GeneralBuilder: ...
    @classmethod
    def make_plain_linear_planner(self) -> Planner: ...
    @classmethod
    def make_plain_greedy_planner(self) -> Planner: ...
    @classmethod
    def make_dp_greedy_planner(self) -> Planner: ...
    @classmethod
    def make_lower(self) -> Lower: ...
    @classmethod
    def make_runner(self) -> Runner: ...

class Flow: ...
class Sequence: ...
class Graph: ...
class Index: ...
class Builder: ...

class GeneralBuilder(Builder):
    @classmethod
    def run(self, input: str, sequence=None, index=None) -> None: ...

class Converter:
    @classmethod
    def run(self, graph: Graph) -> Flow: ...
    @staticmethod
    def make() -> Converter: ...

class Executor:
    @classmethod
    def compile(
        self,
        input: str,
        mlir: Optional[str] = None,
        llvm: Optional[str] = None,
        json: Optional[str] = None,
    ) -> None: ...
    @classmethod
    def invoke(self, args: Dict[str, np.ndarray]) -> int: ...
    @staticmethod
    def make_plain_linear(name: str, epoch: int = 1) -> Executor: ...
    @staticmethod
    def make_plain_greedy(name: str, epoch: int = 1) -> Executor: ...
    @staticmethod
    def make_dp_greedy(name: str, epoch: int = 1) -> Executor: ...

class Lower:
    @classmethod
    def run(self, graph: Graph) -> Tuple[Sequence, Index, Dict]: ...

class Parser:
    @classmethod
    def run(self, input: str) -> Graph: ...
    @staticmethod
    def make() -> Parser: ...

class Planner:
    @classmethod
    def run(self, graph: Graph) -> Sequence: ...

class Runner:
    @classmethod
    def run(self, args: Dict[str, np.ndarray], epoch: int = 1) -> int: ...
