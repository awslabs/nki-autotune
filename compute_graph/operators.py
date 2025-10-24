from typing import Dict, List, Tuple

from compute_graph.primitives import AXES_LOC


class Operator:
    """Defines a compute operation in the workload."""

    def __init__(self, op: str, src: List[str], dest: str, params: Dict = {}) -> None:
        self.op = op
        self.src = src
        self.dest = dest
        self.params = params

    def __repr__(self) -> str:
        params_str = f", params={self.params}" if self.params else ""
        return f"Operator('{self.op}': {self.src} -> {self.dest}{params_str})"


class Node:
    def __init__(self, index: int, node_type: str, src: List[Tuple[str, AXES_LOC]], dest: Tuple[str, AXES_LOC]) -> None:
        """Initialize a computation graph node.

        Args:
            index: Unique node identifier in the graph.
            node_type: Type of node - must be 'load', 'compute', or 'store'.
            src: List of source tensor names and location metadata for source tensors.
            dest: Destination tensor name and location metadata for destination tensor.
        """
        self.index = index
        self.type = node_type
        assert node_type in ["load", "compute", "store"], f"Invalid node type {node_type}"
        self.src = src
        if node_type == "load" or node_type == "store":
            assert len(src) == 1, f"Load and store nodes must have exactly one source, got {len(src)}"
        self.dest = dest

    def __repr__(self) -> str:
        attrs = []
        if hasattr(self, "op"):
            attrs.append(f"op='{self.op}'")
        attrs.append(f"src={self.src}")
        attrs.append(f"dest='{self.dest}'")
        if hasattr(self, "params"):
            attrs.append(f"params={self.params}")
        return f"{self.index}:{self.type}({', '.join(attrs)})"


class LoadNode(Node):
    def __init__(self, index: int, src: Tuple[str, AXES_LOC], dest: Tuple[str, AXES_LOC]) -> None:
        super().__init__(index=index, node_type="load", src=[src], dest=dest)


class ComputeNode(Node):
    def __init__(
        self, index: int, op: str, src: List[Tuple[str, AXES_LOC]], params: Dict, dest: Tuple[str, AXES_LOC]
    ) -> None:
        super().__init__(index=index, node_type="compute", src=src, dest=dest)
        self.op = op
        self.params = params


class StoreNode(Node):
    def __init__(self, index: int, src: Tuple[str, AXES_LOC], dest: Tuple[str, AXES_LOC]) -> None:
        super().__init__(index=index, node_type="store", src=[src], dest=dest)
