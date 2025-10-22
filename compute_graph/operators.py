from typing import Dict, List


class Operator:
    """Defines a compute operation in the workload."""

    def __init__(self, op_type: str, inputs: List[str], output: str, params: Dict = {}) -> None:
        self.op_type = op_type
        self.inputs = inputs
        self.output = output
        self.params = params

    def __repr__(self) -> str:
        params_str = f", params={self.params}" if self.params else ""
        return f"Operator('{self.op_type}': {self.inputs} -> {self.output}{params_str})"
