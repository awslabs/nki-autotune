from typing import Dict, List, Tuple


class Operator:
    """Defines a compute operation in the workload."""

    def __init__(self, op: str, src: List[str], output: str, params: Dict = {}) -> None:
        self.op = op
        self.src = src
        self.output = output
        self.params = params

    def __repr__(self) -> str:
        params_str = f", params={self.params}" if self.params else ""
        return f"Operator('{self.op}': {self.src} -> {self.output}{params_str})"


class Node:
    def __init__(self, index: int, node_type: str, src: List[str], dest: str) -> None:
        self.index = index
        self.type = node_type
        self.src = src
        self.dest = dest

    def __repr__(self) -> str:
        return f"{self.index}_Node(type='{self.type}')"


class LoadNode(Node):
    def __init__(self, index: int, src: str, indices: Dict[int, Tuple[int, int]], dest: str) -> None:
        super().__init__(index, "load", [src], dest)
        self.indices = indices

    def __repr__(self) -> str:
        return f"{self.index}:Load(" f"src='{self.src[0]}', " f"indices={self.indices}, " f"dest='{self.dest}')"


class ComputeNode(Node):
    def __init__(self, index: int, op: str, src: List[str], params: Dict, dest: str) -> None:
        super().__init__(index, "compute", src, dest)
        self.op = op
        self.params = params

    def __repr__(self) -> str:
        return f"{self.index}:Compute(" f"op='{self.op}', " f"src={self.src}, " f"dest='{self.dest}')"


class StoreNode(Node):
    def __init__(self, index: int, src: str, indices: Dict[int, Tuple[int, int]], dest: str) -> None:
        super().__init__(index, "store", [src], dest)
        self.indices = indices

    def __repr__(self) -> str:
        return f"{self.index}:Store(" f"src='{self.src[0]}', " f"indices={self.indices}, " f"dest='{self.dest}')"
