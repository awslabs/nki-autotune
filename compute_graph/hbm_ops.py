from compute_graph.buffer_tensor import BufferTensor
from compute_graph.hbm_tensor import HBMTensor


class Load:
    """
    Load operator.
    """

    def __init__(self, dest: BufferTensor, src: HBMTensor) -> None:
        self.dest = dest
        self.src = src
        assert dest.buffer == "SBUF", f"Cannot load {dest} from HBM"

    def __repr__(self) -> str:
        return f"{self.dest} = Load(src={self.src})"


class Store:
    """
    Store operator.
    """

    def __init__(self, dest: HBMTensor, value: BufferTensor) -> None:
        self.dest = dest
        self.value = value
        assert value.buffer == "SBUF", f"Cannot store {value} to HBM"

    def __repr__(self) -> str:
        return f"Store(dst={self.dest}, value={self.value})"
