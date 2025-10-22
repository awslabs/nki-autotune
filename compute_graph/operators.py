from typing import Dict, List


class Operator:
    """Defines a compute operation in the workload."""

    def __init__(self, op_type: str, inputs: List[str], outputs: List[str], params: Dict = {}) -> None:
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.params = params

    def __repr__(self) -> str:
        params_str = f", params={self.params}" if self.params else ""
        return f"Operator('{self.op_type}': {self.inputs} -> {self.outputs}{params_str})"
