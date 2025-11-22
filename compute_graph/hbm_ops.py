from compute_graph.tensors import Axis, BufferTensor, HBMTensor


class Load:
    """
    Load operator.
    """

    def __init__(self, dest: str, dest_axes: tuple[Axis, ...], src: str, src_axes: tuple[Axis, ...]) -> None:
        self.dest = dest
        self.dest_axes = dest_axes
        self.src = src
        self.src_axes = src_axes

    def __repr__(self) -> str:
        code = f"{self.dest}{list(self.dest_axes)} = nl.load(src={self.src}{list(self.src_axes)})"
        return code


class Store:
    """
    Store operator.
    """

    def __init__(self, dst: HBMTensor, value: BufferTensor) -> None:
        self.dst = dst
        self.value = value

    def __repr__(self) -> str:
        code = f"nl.store({self.dst}, value={self.value})"
        return code
