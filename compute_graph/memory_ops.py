from compute_graph.buffer_tensor import BufferTensor
from compute_graph.hbm_tensor import HBMTensor


class MemoryOp:
    """Base class for memory operations."""

    def codegen(self) -> str:
        """Generate NKI code for this memory operation."""
        raise NotImplementedError(f"codegen not implemented for {type(self).__name__}")


class Load(MemoryOp):
    """Load data from HBM to SBUF."""

    def __init__(self, dest: BufferTensor, src: HBMTensor) -> None:
        self.dest = dest
        self.src = src
        assert dest.buffer == "SBUF", f"Cannot load {dest} from HBM"

    def codegen(self) -> str:
        """Generate nl.load from HBM to SBUF."""
        indices = _generate_hbm_indices(self.src)
        return f"{self.dest.name} = nl.load({self.src.name}[{indices}])"

    def __repr__(self) -> str:
        return f"{self.dest} = Load(src={self.src})"


class Store(MemoryOp):
    """Store data from SBUF to HBM."""

    def __init__(self, dest: HBMTensor, value: BufferTensor) -> None:
        self.dest = dest
        self.value = value
        assert value.buffer == "SBUF", f"Cannot store {value} to HBM"

    def codegen(self) -> str:
        """Generate nl.store from SBUF to HBM."""
        indices = _generate_hbm_indices(self.dest)
        return f"nl.store({self.dest.name}[{indices}], value={self.value.name})"

    def __repr__(self) -> str:
        return f"Store(dst={self.dest}, value={self.value})"


class Allocate(MemoryOp):
    """Allocate a tensor in on-chip memory.

    Creates nl.ndarray with specified shape, and buffer location.
    """

    def __init__(self, tensor: BufferTensor) -> None:
        self.tensor = tensor

    def codegen(self) -> str:
        """Generate nl.ndarray allocation."""
        buffer_map = {"SBUF": "nl.sbuf", "PSUM": "nl.psum"}
        buffer = buffer_map.get(self.tensor.buffer, "nl.sbuf")
        return f"{self.tensor.name} = nl.ndarray({self.tensor.shape}, dtype=nl.float32, buffer={buffer})"

    def __repr__(self) -> str:
        return f"{self.tensor.name} = Allocate(shape={self.tensor.shape}, buffer={self.tensor.buffer})"


def _generate_hbm_indices(tensor: HBMTensor) -> str:
    """Generate index expressions for HBM tensor access.

    For each axis, generate a slice from start_tile*tile_size to end_tile*tile_size.
    """
    indices = []
    for axis in tensor.axes:
        start = axis.start_tile * axis.tile_size
        end = axis.end_tile * axis.tile_size
        indices.append(f"{start}:{end}")
    return ", ".join(indices)
