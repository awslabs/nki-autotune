from typing import Dict, List, Tuple


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


class Node:
    def __init__(self, index: int, node_type: str, dest: str) -> None:
        self.index = index
        self.type = node_type
        self.dest = dest

    def __repr__(self) -> str:
        return f"Node(index={self.index}, type='{self.type}')"


class LoadNode(Node):
    def __init__(self, index: int, input_tensor: str, load_indices: Dict[int, Tuple[int, int]], dest: str) -> None:
        super().__init__(index, "load", dest)
        self.input_tensor = input_tensor
        self.load_indices = load_indices

    def __repr__(self) -> str:
        return (
            f"LoadNode("
            f"index={self.index}, "
            f"input_tensor='{self.input_tensor}', "
            f"load_indices={self.load_indices}, "
            f"dest='{self.dest}')"
        )


class ComputeNode(Node):
    def __init__(self, index: int, op_type: str, inputs: List[str], params: Dict, dest: str) -> None:
        super().__init__(index, "compute", dest)
        self.op_type = op_type
        self.inputs = inputs
        self.params = params

    def __repr__(self) -> str:
        return (
            f"ComputeNode("
            f"index={self.index}, "
            f"op_type='{self.op_type}', "
            f"inputs={self.inputs}, "
            f"dest='{self.dest}')"
        )


class StoreNode(Node):
    def __init__(self, index: int, src_tensor: str, store_indices: Dict[int, Tuple[int, int]], dest: str) -> None:
        super().__init__(index, "store", dest)
        self.src_tensor = src_tensor
        self.store_indices = store_indices

    def __repr__(self) -> str:
        return (
            f"StoreNode("
            f"index={self.index}, "
            f"src_tensor='{self.src_tensor}', "
            f"store_indices={self.store_indices}, "
            f"dest='{self.dest}')"
        )
