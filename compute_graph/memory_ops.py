from compute_graph.buffer_tensor import BufferTensor
from compute_graph.hbm_tensor import HBMTensor


class MemoryOp:
    """Base class for memory operations."""


class Load(MemoryOp):
    """
    Load operator.
    """

    def __init__(self, dest: BufferTensor, src: HBMTensor) -> None:
        self.dest = dest
        self.src = src
        assert dest.buffer == "SBUF", f"Cannot load {dest} from HBM"

    def __repr__(self) -> str:
        return f"{self.dest} = Load(src={self.src})"


class Store(MemoryOp):
    """
    Store operator.
    """

    def __init__(self, dest: HBMTensor, value: BufferTensor) -> None:
        self.dest = dest
        self.value = value
        assert value.buffer == "SBUF", f"Cannot store {value} to HBM"

    def __repr__(self) -> str:
        return f"Store(dst={self.dest}, value={self.value})"


class Allocate(MemoryOp):
    """Allocate a tensor in on-chip memory.

    Creates nl.ndarray with specified shape, and buffer location.
    """

    def __init__(self, tensor: BufferTensor) -> None:
        self.tensor = tensor

    def __repr__(self) -> str:
        return f"{self.tensor.name} = Allocate(shape={self.tensor.shape}, buffer={self.tensor.buffer})"
