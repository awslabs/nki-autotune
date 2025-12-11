from compute_graph.buffer_tensor import BufferTensor
from compute_graph.node.hbm_tensor import HBMTensor
from compute_graph.node.node import Node


class Load(Node):
    """Load data from HBM to SBUF."""

    def __init__(self, dest: str, src: str) -> None:
        super().__init__(
            read_args=("src",),
            write_args=("dest",),
            arg_to_var={"src": src, "dest": dest},
            arg_to_axes={"src": ("P", "F"), "dest": ("P", "F")},
        )

    def codegen(self) -> str:
        """Generate nl.load from HBM to SBUF."""
        hbm_tensor = self.arg_to_tensor["src"]
        assert isinstance(hbm_tensor, HBMTensor)
        indices = _generate_hbm_indices(hbm_tensor)
        buffer_tensor = self.arg_to_tensor["dest"]
        assert isinstance(buffer_tensor, BufferTensor)
        return f"nisa.dma_copy(dst={buffer_tensor.name}, src={hbm_tensor.name}[{indices}])"

    def __repr__(self) -> str:
        return f"{self._format_tensor('dest')} = Load(src={self._format_tensor('src')})"


class Store(Node):
    """Store data from SBUF to HBM."""

    def __init__(self, dest: str, value: str) -> None:
        super().__init__(
            read_args=("value",),
            write_args=("dest",),
            arg_to_var={"value": value, "dest": dest},
            arg_to_axes={"value": ("P", "F"), "dest": ("P", "F")},
        )

    def codegen(self) -> str:
        """Generate nl.store from SBUF to HBM."""
        hbm_tensor = self.arg_to_tensor["dest"]
        assert isinstance(hbm_tensor, HBMTensor)
        indices = _generate_hbm_indices(hbm_tensor)
        buffer_tensor = self.arg_to_tensor["value"]
        assert isinstance(buffer_tensor, BufferTensor)
        return f"nl.store({hbm_tensor.name}[{indices}], value={buffer_tensor.name})"

    def __repr__(self) -> str:
        return f"Store(dest={self._format_tensor('dest')}, value={self._format_tensor('value')})"


class Allocate(Node):
    """Allocate a tensor in on-chip memory.

    Creates nl.ndarray with specified shape, and buffer location.
    """

    def __init__(self, tensor: str, buffer: str) -> None:
        super().__init__(
            read_args=(), write_args=("tensor",), arg_to_var={"tensor": tensor}, arg_to_axes={"tensor": ("P", "F")}
        )
        self.buffer = buffer
        assert buffer in ["SBUF", "PSUM"], f"Illegal buffer type {buffer}"

    def codegen(self) -> str:
        """Generate nl.ndarray allocation."""
        tensor = self.arg_to_tensor["tensor"]
        assert isinstance(tensor, BufferTensor)
        buffer_name = "nl.sbuf" if self.buffer == "SBUF" else "nl.psum"
        return f"{tensor.name} = nl.ndarray({tensor.shape}, dtype=nl.float32, buffer={buffer_name})"

    def __repr__(self) -> str:
        return f"{self._format_tensor('tensor')} = Allocate(buffer={self.buffer})"


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
