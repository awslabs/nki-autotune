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
1. Subclass NkiOp and implement all abstract methods
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

    Inside this context, NkiOp calls will perform tracing (returning TracedTensor)
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


class NkiOp(ABC):
    """Base class for NKI custom operators.

    Subclasses implement hardware-native operations that integrate with
    TracedTensor for dimension tracking and code generation.

    Dispatches based on tracing context:
    - Inside tracing_enabled() -> tracing (returns TracedTensor)
    - Outside tracing context -> simulation (returns numpy array)

    Attributes:
        op_name: Name of the operation (e.g., "nc_matmul").
    """

    op_name: str

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


class NKIMatmul(NkiOp):
    """NKI matrix multiplication following nc_matmul semantics.

    Expects inputs in KM x KN layout where K is the partition dimension:
    - lhs (stationary): [K, M] where K <= 128 (pmax), M <= 128 (stationary fmax)
    - rhs (moving): [K, N] where K <= 128 (pmax), N <= 512 (moving fmax)
    - result: [M, N]

    The K dimension is contracted (unified) during the operation.

    Example:
        >>> nc_matmul = NKIMatmul()
        >>> lhs = TracedTensor("lhs", (128, 64), ["K", "M"], tracker)
        >>> rhs = TracedTensor("rhs", (128, 256), ["K", "N"], tracker)
        >>> result = nc_matmul(lhs, rhs)  # shape (64, 256), dims ["M", "N"]
    """

    op_name = "nc_matmul"

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


OP_REGISTRY: dict[str, NkiOp] = {"nc_matmul": NKIMatmul()}
"""Registry mapping operation names to NkiOp instances.

To add a new operator, create a subclass of NkiOp and add an instance here.
"""


def ndarray(shape: tuple[int, ...], dtype: type = np.float32) -> NDArray:
    """Allocate a tensor.

    In simulation mode, returns a numpy array. When lowered to NKI,
    becomes nl.ndarray allocation.

    Args:
        shape: Shape of the tensor.
        dtype: Data type (default np.float32).

    Returns:
        Allocated numpy array.
    """
    return np.empty(shape, dtype=dtype)
