from compute_graph.tensors import HBMTensor, TensorBuffer


class Node:
    def __init__(self, index: int, node_type: str) -> None:
        self.index = index
        self.type = node_type
        assert node_type in ["load", "compute", "store"], f"Invalid node type {node_type}"


class LoadNode(Node):
    def __init__(self, index: int, src: HBMTensor, dest: TensorBuffer) -> None:
        super().__init__(index=index, node_type="load")
        self.src = src
        self.dest = dest

    def __repr__(self) -> str:
        return f"{self.index}:load({self.src} -> {self.dest})"


class ComputeNode(Node):
    def __init__(self, index: int, op: str, src: list[TensorBuffer], dest: TensorBuffer, params: dict) -> None:
        super().__init__(index=index, node_type="compute")
        self.op = op
        self.src = src
        self.dest = dest
        self.params = params


class StoreNode(Node):
    def __init__(self, index: int, src: TensorBuffer, dest: HBMTensor) -> None:
        super().__init__(index=index, node_type="store")
        self.src = src
        self.dest = dest
