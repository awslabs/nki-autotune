from typing import Dict, List, Tuple


class Operator:
    """Defines a compute operation in the workload."""

    def __init__(self, op_type: str, inputs: List[str], params: Dict) -> None:
        self.op_type = op_type
        self.inputs = inputs
        self.params = params

    def __repr__(self) -> str:
        return f"Operator(op_type={self.op_type}, inputs={self.inputs}, params={self.params})"


class Workload:
    """High-level workload specification for compute graph generation."""

    def __init__(
        self,
        input_tensors: Dict[str, Tuple[int, ...]],
        operators: List[Operator],
        parallel_axes: List[Tuple[str, int, int]],
        output_tensor: str,
    ) -> None:
        self.input_tensors = input_tensors
        self.operators = operators
        self.parallel_axes = parallel_axes
        self.output_tensor = output_tensor

    def __repr__(self) -> str:
        return (
            f"Workload(input_tensors={self.input_tensors}, "
            f"operators={self.operators}, "
            f"parallel_axes={self.parallel_axes}, "
            f"output_tensor={self.output_tensor})"
        )
