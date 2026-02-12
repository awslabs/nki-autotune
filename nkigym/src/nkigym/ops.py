"""NKI operator definitions.

This module provides base classes and implementations for NKI (Neuron Kernel Interface)
operators that can be used with TracedTensor for symbolic tracing.

NKI operators follow hardware-native semantics:
- nc_matmul expects KM x KN layout (partition dimension first)
- Dimension tracking integrates with the existing _DimTracker

Tracing is enabled via context manager:
- Normal calls return numpy arrays (simulation)
- Inside tracing_enabled() context, calls return TracedTensor

To add a new operator:
1. Subclass NKIOp and implement all abstract methods
2. Register an instance in OP_REGISTRY with the operator name as key
"""

import threading
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from nkigym.tiling.tensor import TracedTensor


_tracing_context = threading.local()


def is_tracing() -> bool:
    """Check if tracing mode is currently enabled.

    Returns:
        True if inside a tracing_enabled() context, False otherwise.
    """
    return getattr(_tracing_context, "enabled", False)


@contextmanager
def tracing_enabled() -> Generator[None, None, None]:
    """Context manager to enable tracing mode.

    Inside this context, NKIOp calls will perform tracing (returning TracedTensor)
    instead of simulation (returning numpy arrays).

    Example:
        with tracing_enabled():
            result = func(**traced_tensors)  # Returns TracedTensor
    """
    _tracing_context.enabled = True
    try:
        yield
    finally:
        _tracing_context.enabled = False


class NKIOp(ABC):
    """Base class for NKI custom operators.

    Subclasses implement hardware-native operations that integrate with
    TracedTensor for dimension tracking and code generation.

    Dispatches based on tracing context:
    - Inside tracing_enabled() -> tracing (returns TracedTensor)
    - Outside tracing context -> simulation (returns numpy array)

    Attributes:
        op_name: Name of the operation (e.g., "nc_matmul").
        operand_names: Positional names for each operand slot.
        read_positions: Indices into operand_names that are read by this op.
        write_positions: Indices into operand_names that are written by this op.
        tile_limits: Hardware tile size limits per dimension name.
    """

    op_name: str
    operand_names: tuple[str, ...] = ()
    read_positions: tuple[int, ...] = ()
    write_positions: tuple[int, ...] = ()
    tile_limits: dict[str, int] = {}

    def can_merge_dim(self, dim: int, new_size: int) -> bool:
        """Check whether merging along a dimension respects hardware limits.

        Args:
            dim: Dimension index in the operand.
            new_size: Proposed merged size for that dimension.

        Returns:
            True if the merged size is within the tile limit for that dimension,
            or if the op has no tile limits.
        """
        if not self.tile_limits:
            return True
        dim_names = list(self.tile_limits.keys())
        if dim >= len(dim_names):
            return True
        limit = self.tile_limits[dim_names[dim]]
        return new_size <= limit

    def can_merge_operand_dim(self, operand_idx: int, dim: int, new_size: int) -> bool:
        """Check whether merging a specific operand's dimension respects limits.

        Subclasses override this to provide operand-aware dimension mapping.
        Default delegates to ``can_merge_dim(dim, new_size)``.

        Args:
            operand_idx: Index of the input operand (0-based, excludes output).
            dim: Dimension index within the operand.
            new_size: Proposed merged size.

        Returns:
            True if the merged size is within the tile limit.
        """
        return self.can_merge_dim(dim, new_size)

    def __call__(self, *args: NDArray) -> NDArray:
        """Execute the operation.

        In normal usage, performs numpy simulation. Inside tracing_enabled()
        context, performs tracing on TracedTensors.

        Args:
            *args: Input numpy arrays (or TracedTensors when tracing).

        Returns:
            Numpy array result (or TracedTensor when tracing).
        """
        if is_tracing():
            return self._trace(*args)  # type: ignore[arg-type, return-value]
        else:
            return self.simulate(*args)

    @abstractmethod
    def _trace(self, *args: "TracedTensor") -> "TracedTensor":
        """Execute tracing on TracedTensors.

        Args:
            *args: Input TracedTensors.

        Returns:
            TracedTensor result with tracked dimensions.
        """

    @abstractmethod
    def generate_nki(self, inputs: list[str], output: str) -> str:
        """Generate NKI code for this operation.

        Args:
            inputs: List of input variable names.
            output: Output variable name.

        Returns:
            NKI code string.
        """

    @abstractmethod
    def simulate(self, *args: NDArray) -> NDArray:
        """Execute the numpy equivalent for simulation and correctness checking.

        Args:
            *args: Input numpy arrays.

        Returns:
            Numpy array result matching NKI operation semantics.
        """

    @abstractmethod
    def generate_expr(self, inputs: list[str]) -> str:
        """Generate numpy expression for tiled code generation.

        Args:
            inputs: List of input variable names.

        Returns:
            Numpy expression string (e.g., "nkigym.nc_matmul(a, b)").
        """

    @abstractmethod
    def reduce(self, result_var: str, inputs: list[str]) -> str | None:
        """Generate accumulation expression for reduction tiling.

        Args:
            result_var: Name of the accumulator variable.
            inputs: List of input variable names.

        Returns:
            In-place accumulation expression, or None if op doesn't perform reduction.
        """

    @abstractmethod
    def output_shape(self, input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        """Compute the output tensor shape from input tensor shapes.

        Args:
            input_shapes: List of input tensor shapes.

        Returns:
            Shape tuple of the output tensor.
        """


class LoadOp(NKIOp):
    """Load operation: copies a slice from an input tensor to a local tile.

    Operands: (src, dst) where src is the input tensor slice and dst is the
    local tile variable.

    Attributes:
        op_name: "load"
        operand_names: ("src", "dst")
        read_positions: (0,) -- reads from src
        write_positions: (1,) -- writes to dst
    """

    op_name = "load"
    operand_names = ("src", "dst")
    read_positions = (0,)
    write_positions = (1,)
    tile_limits: dict[str, int] = {}

    def _trace(self, *args: "TracedTensor") -> "TracedTensor":
        """Not used for LoadOp."""
        raise NotImplementedError("LoadOp does not support tracing")

    def generate_nki(self, inputs: list[str], output: str) -> str:
        """Not used for LoadOp."""
        raise NotImplementedError("LoadOp uses program-level codegen")

    def simulate(self, *args: NDArray) -> NDArray:
        """Not used for LoadOp."""
        raise NotImplementedError("LoadOp does not support simulation")

    def generate_expr(self, inputs: list[str]) -> str:
        """Not used for LoadOp."""
        raise NotImplementedError("LoadOp uses program-level codegen")

    def reduce(self, result_var: str, inputs: list[str]) -> str | None:
        """LoadOp does not perform reduction."""
        return None

    def output_shape(self, input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        """Not used for LoadOp."""
        raise NotImplementedError("LoadOp does not compute output shapes")


class StoreOp(NKIOp):
    """Store operation: copies a local tile to an output tensor slice.

    Operands: (src, dst) where src is the local tile and dst is the
    output tensor slice.

    Attributes:
        op_name: "store"
        operand_names: ("src", "dst")
        read_positions: (0,) -- reads from src
        write_positions: (1,) -- writes to dst
    """

    op_name = "store"
    operand_names = ("src", "dst")
    read_positions = (0,)
    write_positions = (1,)
    tile_limits: dict[str, int] = {}

    def _trace(self, *args: "TracedTensor") -> "TracedTensor":
        """Not used for StoreOp."""
        raise NotImplementedError("StoreOp does not support tracing")

    def generate_nki(self, inputs: list[str], output: str) -> str:
        """Not used for StoreOp."""
        raise NotImplementedError("StoreOp uses program-level codegen")

    def simulate(self, *args: NDArray) -> NDArray:
        """Not used for StoreOp."""
        raise NotImplementedError("StoreOp does not support simulation")

    def generate_expr(self, inputs: list[str]) -> str:
        """Not used for StoreOp."""
        raise NotImplementedError("StoreOp uses program-level codegen")

    def reduce(self, result_var: str, inputs: list[str]) -> str | None:
        """StoreOp does not perform reduction."""
        return None

    def output_shape(self, input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        """Not used for StoreOp."""
        raise NotImplementedError("StoreOp does not compute output shapes")


class AllocOp(NKIOp):
    """Allocation operation: allocates an output tensor.

    Per-dtype singletons keep the statement uniformly (op, operands) with no
    extra metadata. Shape is derivable from the operand's slices.

    Operands: (tensor,) where tensor is the allocated variable with
    full-range slices encoding the shape.

    Attributes:
        op_name: "alloc_<dtype>" (e.g., "alloc_float32").
        operand_names: ("tensor",)
        read_positions: () -- no reads
        write_positions: (0,) -- writes the allocation
        dtype: Numpy dtype instance.
    """

    operand_names = ("tensor",)
    read_positions = ()
    write_positions = (0,)
    tile_limits: dict[str, int] = {}

    def __init__(self, dtype: type) -> None:
        """Initialize an AllocOp for a specific dtype.

        Args:
            dtype: Numpy dtype (e.g., np.float32, np.float64).
        """
        self.dtype = dtype
        self.op_name = f"alloc_{np.dtype(dtype).name}"

    def _trace(self, *args: "TracedTensor") -> "TracedTensor":
        """Not used for AllocOp."""
        raise NotImplementedError("AllocOp does not support tracing")

    def generate_nki(self, inputs: list[str], output: str) -> str:
        """Not used for AllocOp."""
        raise NotImplementedError("AllocOp uses program-level codegen")

    def simulate(self, *args: NDArray) -> NDArray:
        """Not used for AllocOp."""
        raise NotImplementedError("AllocOp does not support simulation")

    def generate_expr(self, inputs: list[str]) -> str:
        """Not used for AllocOp."""
        raise NotImplementedError("AllocOp uses program-level codegen")

    def reduce(self, result_var: str, inputs: list[str]) -> str | None:
        """AllocOp does not perform reduction."""
        return None

    def output_shape(self, input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        """Not used for AllocOp."""
        raise NotImplementedError("AllocOp does not compute output shapes")


class NKIMatmul(NKIOp):
    """NKI matrix multiplication following nc_matmul semantics.

    Expects inputs in KM x KN layout where K is the partition dimension:
    - lhs (stationary): [K, M] where K <= 128 (pmax), M <= 128 (stationary fmax)
    - rhs (moving): [K, N] where K <= 128 (pmax), N <= 512 (moving fmax)
    - result: [M, N]

    The K dimension is contracted (unified) during the operation.

    Attributes:
        operand_names: ("lhs", "rhs", "dst")
        read_positions: (0, 1) -- reads lhs and rhs
        write_positions: (2,) -- writes dst
        tile_limits: {"M": 128, "K": 128, "N": 512}

    Example:
        >>> nc_matmul = NKIMatmul()
        >>> lhs = TracedTensor("lhs", (128, 64), ["K", "M"], tracker)
        >>> rhs = TracedTensor("rhs", (128, 256), ["K", "N"], tracker)
        >>> result = nc_matmul(lhs, rhs)  # shape (64, 256), dims ["M", "N"]
    """

    op_name = "nc_matmul"
    operand_names = ("lhs", "rhs", "dst")
    read_positions = (0, 1)
    write_positions = (2,)
    tile_limits = {"M": 128, "K": 128, "N": 512}

    _operand_dim_names: tuple[tuple[str, ...], ...] = (("K", "M"), ("K", "N"))

    def can_merge_operand_dim(self, operand_idx: int, dim: int, new_size: int) -> bool:
        """Check tile limit for a specific operand dimension of nc_matmul.

        Maps (operand_idx, dim) to the abstract dimension name (M, K, N)
        and checks against the corresponding tile limit.

        Args:
            operand_idx: 0 for lhs [K, M], 1 for rhs [K, N].
            dim: Dimension within the operand.
            new_size: Proposed merged size.

        Returns:
            True if within the tile limit.
        """
        if operand_idx < len(self._operand_dim_names):
            dim_names = self._operand_dim_names[operand_idx]
            if dim < len(dim_names):
                dim_name = dim_names[dim]
                limit = self.tile_limits.get(dim_name)
                if limit is not None:
                    return new_size <= limit
        return True

    def _trace(self, lhs: "TracedTensor", rhs: "TracedTensor") -> "TracedTensor":
        """Execute tracing for nc_matmul on TracedTensors.

        Contracts K dimension (axis 0 of both inputs):
        lhs[K, M] @ rhs[K, N] -> result[M, N]

        Args:
            lhs: Left-hand side (stationary) tensor of shape [K, M].
            rhs: Right-hand side (moving) tensor of shape [K, N].

        Returns:
            TracedTensor of shape [M, N] with dims [lhs.dims[1], rhs.dims[1]].

        Raises:
            ValueError: If inputs don't have exactly 2 dimensions.
        """
        from nkigym.tiling.tensor import TracedTensor

        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            raise ValueError(f"nc_matmul requires 2D tensors, got shapes {lhs.shape} and {rhs.shape}")

        lhs.tracker.unify(lhs.dims[0], rhs.dims[0])

        result_dims = [lhs.dims[1], rhs.dims[1]]
        result_shape = (lhs.shape[1], rhs.shape[1])

        output_name = lhs.tracker.new_intermediate_name()
        lhs.tracker.record_op(self.op_name, [lhs.name, rhs.name], output_name)

        return TracedTensor(name=output_name, shape=result_shape, dims=result_dims, tracker=lhs.tracker)

    def generate_nki(self, inputs: list[str], output: str) -> str:
        """Generate NKI code for nc_matmul.

        Args:
            inputs: List of input variable names [lhs, rhs].
            output: Output variable name.

        Returns:
            NKI code string calling nisa.nc_matmul with pre-allocated PSUM buffer.
        """
        alloc = f'{output} = nl.zeros(({inputs[0]}.shape[1], {inputs[1]}.shape[1]), dtype={inputs[0]}.dtype, buffer=nl.psum, name="{output}")'
        matmul = f"nisa.nc_matmul({output}, {inputs[0]}, {inputs[1]})"
        return f"{alloc}\n{matmul}"

    def simulate(self, lhs: NDArray, rhs: NDArray) -> NDArray:
        """Execute numpy equivalent of nc_matmul.

        Computes lhs.T @ rhs where:
        - lhs: [K, M] -> lhs.T: [M, K]
        - rhs: [K, N]
        - result: [M, N]

        Args:
            lhs: Left-hand side array of shape [K, M].
            rhs: Right-hand side array of shape [K, N].

        Returns:
            Result array of shape [M, N].
        """
        return np.matmul(lhs.T, rhs)

    def generate_expr(self, inputs: list[str]) -> str:
        """Generate nkigym expression for nc_matmul.

        Args:
            inputs: List of input variable names [lhs, rhs].

        Returns:
            nkigym.nc_matmul expression.
        """
        return f"nkigym.nc_matmul({inputs[0]}, {inputs[1]})"

    def reduce(self, result_var: str, inputs: list[str]) -> str:
        """Generate accumulation expression for reduction tiling.

        Args:
            result_var: Name of the accumulator variable.
            inputs: List of input variable names [lhs, rhs].

        Returns:
            In-place addition expression.
        """
        return f"{result_var} += nkigym.nc_matmul({inputs[0]}, {inputs[1]})"

    def output_shape(self, input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        """Compute output shape for nc_matmul: [K,M] @ [K,N] -> [M,N].

        Args:
            input_shapes: List of [lhs_shape, rhs_shape].

        Returns:
            Shape tuple (M, N).
        """
        return (input_shapes[0][1], input_shapes[1][1])


class ElementwiseOp(NKIOp):
    """Elementwise operation with keyword arguments.

    Handles operations like tensor_tensor, activation, and tensor_scalar
    that take tensor inputs plus keyword arguments specifying the operation.
    Output shape matches the first input's shape.

    Instances are parameterized by kwargs and compare equal if kwargs match.

    Attributes:
        op_name: Operation name (e.g., "tensor_tensor").
        kwargs_repr: Sorted tuple of (key, repr_string) for kwargs.
    """

    read_positions = ()
    write_positions = ()
    tile_limits: dict[str, int] = {}

    def __init__(self, op_name: str, kwargs_repr: tuple[tuple[str, str], ...] = ()) -> None:
        """Initialize an ElementwiseOp.

        Args:
            op_name: Operation name (e.g., "tensor_tensor").
            kwargs_repr: Sorted tuple of (key, repr_string) for kwargs.
        """
        self.op_name = op_name
        self.kwargs_repr = kwargs_repr

    def __eq__(self, other: object) -> bool:
        """Check equality based on op_name and kwargs."""
        if not isinstance(other, ElementwiseOp):
            return NotImplemented
        return self.op_name == other.op_name and self.kwargs_repr == other.kwargs_repr

    def __hash__(self) -> int:
        """Hash based on op_name and kwargs."""
        return hash((self.op_name, self.kwargs_repr))

    def _trace(self, *args: "TracedTensor") -> "TracedTensor":
        """Not used for ElementwiseOp."""
        raise NotImplementedError("ElementwiseOp does not support tracing")

    def generate_nki(self, inputs: list[str], output: str) -> str:
        """Not used for ElementwiseOp."""
        raise NotImplementedError("ElementwiseOp uses program-level codegen")

    def simulate(self, *args: NDArray) -> NDArray:
        """Not used for ElementwiseOp."""
        raise NotImplementedError("ElementwiseOp does not support simulation")

    def generate_expr(self, inputs: list[str]) -> str:
        """Not used for ElementwiseOp."""
        raise NotImplementedError("ElementwiseOp uses program-level codegen")

    def reduce(self, result_var: str, inputs: list[str]) -> str | None:
        """ElementwiseOp does not perform reduction."""
        return None

    def output_shape(self, input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        """Output shape matches first input shape.

        Args:
            input_shapes: List of input tensor shapes.

        Returns:
            Shape of the first input.
        """
        return input_shapes[0]


LOAD_OP = LoadOp()
STORE_OP = StoreOp()
ALLOC_F32_OP = AllocOp(np.float32)
ALLOC_F64_OP = AllocOp(np.float64)
NC_MATMUL_OP = NKIMatmul()

ALLOC_OPS: dict[str, AllocOp] = {"float32": ALLOC_F32_OP, "float64": ALLOC_F64_OP}

ELEMENTWISE_OP_NAMES: set[str] = {"tensor_tensor", "activation", "tensor_scalar"}

OP_REGISTRY: dict[str, NKIOp] = {
    "nc_matmul": NC_MATMUL_OP,
    "load": LOAD_OP,
    "store": STORE_OP,
    "alloc_float32": ALLOC_F32_OP,
    "alloc_float64": ALLOC_F64_OP,
}


def ndarray(shape: tuple[int, ...], dtype: type) -> NDArray:
    """Allocate a tensor.

    In simulation mode, returns a numpy array. When lowered to NKI,
    becomes nl.ndarray allocation.

    Args:
        shape: Shape of the tensor.
        dtype: Data type for the tensor.

    Returns:
        Allocated numpy array.
    """
    return np.empty(shape, dtype=dtype)
