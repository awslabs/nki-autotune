from compute_graph.buffer_tensor import SBUFTensor
from compute_graph.hbm_tensor import HBMTensor


class Load:
    """
    Load operator.
    """

    def __init__(self, dest: SBUFTensor, src: HBMTensor) -> None:
        self.dest = dest
        self.src = src

    def __repr__(self) -> str:
        code = f"{self.dest} = Load(src={self.src})"
        return code


class Store:
    """
    Store operator.
    """

    def __init__(self, dest: HBMTensor, value: SBUFTensor) -> None:
        self.dest = dest
        self.value = value

    def __repr__(self) -> str:
        code = f"Store(dst={self.dest}, value={self.value})"
        return code
