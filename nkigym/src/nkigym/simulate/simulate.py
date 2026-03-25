"""CPU simulator for rendered NKI kernel source code.

Executes NKI kernels using numpy at float64 precision for
correctness verification without requiring Neuron hardware.

Mock ``nki``, ``nki.language``, and ``nki.isa`` modules replace the
real NKI SDK.  Each ``nisa.*`` function is implemented with numpy
operations so the kernel runs identically on CPU.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np


class _Op:
    """Sentinel for an NKI operation used as dispatch key.

    Identity-based equality — each sentinel is a unique object shared
    between the mock ``nl`` namespace and the dispatch tables.
    """

    def __init__(self, name: str) -> None:
        """Initialize with operation name."""
        self.name = name

    def __repr__(self) -> str:
        """Return nl.<name> representation."""
        return f"nl.{self.name}"


TANH = _Op("tanh")
EXP = _Op("exp")
ADD = _Op("add")
MULTIPLY = _Op("multiply")
SUBTRACT = _Op("subtract")
MAXIMUM = _Op("maximum")
MAX = _Op("max")
SQUARE = _Op("square")
RSQRT = _Op("rsqrt")

_UNARY_FNS: dict[_Op, Any] = {TANH: np.tanh, EXP: np.exp, SQUARE: np.square, RSQRT: lambda x: 1.0 / np.sqrt(x)}

_BINARY_FNS: dict[_Op, Any] = {ADD: np.add, MULTIPLY: np.multiply, SUBTRACT: np.subtract, MAXIMUM: np.maximum}

_REDUCE_METHODS: dict[_Op, str] = {MAX: "max", ADD: "sum"}

_HBM = SimpleNamespace(name="shared_hbm")
_SBUF = SimpleNamespace(name="sbuf")
_PSUM = SimpleNamespace(name="psum")


def _ndarray(shape: tuple[int, ...], dtype: Any, buffer: Any) -> np.ndarray:
    """Allocate a float64 numpy zeros array (mock nl.ndarray)."""
    return np.zeros(shape, dtype=np.float64)


def _nki_jit(func: Any) -> Any:
    """Identity decorator (mock @nki.jit)."""
    return func


def _flatten_2d(arr: np.ndarray) -> np.ndarray:
    """Flatten multi-dim array to (partition_dim, free_dim)."""
    return arr.reshape(arr.shape[0], -1)


def _dma_copy(dst: np.ndarray, src: np.ndarray) -> None:
    """Copy data with reshape between HBM and SBUF."""
    dst[:] = src.reshape(dst.shape)


def _nc_matmul(dst: np.ndarray, stationary: np.ndarray, moving: np.ndarray) -> None:
    """Accumulate matmul into PSUM: dst += stationary.T @ moving.

    stationary is (K, ...) with free dim M.
    moving is (K, ...) with free dim N.
    Result (M, N) is accumulated into dst.
    """
    s2d = _flatten_2d(stationary)
    m2d = _flatten_2d(moving)
    result = s2d.T @ m2d
    dst[:] += result.reshape(dst.shape)


def _tensor_copy(dst: np.ndarray, src: np.ndarray) -> None:
    """Copy data between memory spaces with reshape."""
    dst[:] = src.reshape(dst.shape)


def _activation(dst: np.ndarray, data: np.ndarray, op: _Op, **kwargs: Any) -> None:
    """Apply element-wise unary activation, optionally with reduction.

    When ``reduce_op`` and ``reduce_res`` are given, also reduces the
    activated data across the free axis and accumulates into reduce_res.
    """
    fn = _UNARY_FNS.get(op)
    if fn is None:
        raise ValueError(f"Unknown activation: {op}")
    activated = fn(data)
    dst[:] = activated.reshape(dst.shape)
    reduce_res = kwargs.get("reduce_res")
    if reduce_res is not None:
        reduce_op = kwargs["reduce_op"]
        d2d = _flatten_2d(activated)
        reduced = reduce_op.reduce(d2d, axis=1)
        reduce_res[:] += reduced.reshape(reduce_res.shape)


def _tensor_tensor(dst: np.ndarray, data1: np.ndarray, data2: np.ndarray, op: _Op) -> None:
    """Apply element-wise binary operation on two tiles."""
    fn = _BINARY_FNS.get(op)
    if fn is None:
        raise ValueError(f"Unknown binary op: {op}")
    dst[:] = fn(data1, data2).reshape(dst.shape)


def _tensor_scalar(dst: np.ndarray, data: np.ndarray, operand0: Any, op0: _Op, **kwargs: Any) -> None:
    """Apply element-wise op between tile and scalar/column vector.

    operand0 may be a lower-dimensional array (broadcast via padding)
    or a scalar literal.  Supports compound ``op1``/``operand1``.
    """
    expanded = operand0
    if isinstance(operand0, np.ndarray) and data.ndim > operand0.ndim:
        pad = data.ndim - operand0.ndim
        expanded = operand0.reshape(operand0.shape + (1,) * pad)
    fn = _BINARY_FNS.get(op0)
    if fn is None:
        raise ValueError(f"Unknown scalar op: {op0}")
    result = fn(data, expanded)
    op1 = kwargs.get("op1")
    if op1 is not None:
        fn1 = _BINARY_FNS.get(op1)
        if fn1 is None:
            raise ValueError(f"Unknown scalar op1: {op1}")
        result = fn1(result, kwargs["operand1"])
    dst[:] = result.reshape(dst.shape)


def _tensor_reduce(dst: np.ndarray, data: np.ndarray, op: _Op) -> None:
    """Reduce along free dimensions, keeping partition dim.

    Flattens data to (partition, free), reduces along axis=1,
    and reshapes result to match dst.
    """
    method = _REDUCE_METHODS.get(op)
    if method is None:
        raise ValueError(f"Unknown reduce op: {op}")
    d2d = _flatten_2d(data)
    reduced = getattr(d2d, method)(axis=1)
    dst[:] = reduced.reshape(dst.shape)


def _nc_transpose(dst: np.ndarray, data: np.ndarray) -> None:
    """PE array transpose: swap partition and free dims.

    Flattens to 2D, transposes, and reshapes back to original
    multi-dim shape.  Correct for square tiles; approximate for
    non-square (sufficient for rmsnorm_matmul where tiles are 128x128).
    """
    d2d = _flatten_2d(data)
    transposed = d2d.T.copy()
    dst[:] = transposed.reshape(dst.shape)


def _nl_copy(src: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Copy data between memory spaces (mock nl.copy)."""
    return src.copy()


def _build_nl() -> SimpleNamespace:
    """Build mock nki.language namespace with all NKI symbols."""
    return SimpleNamespace(
        ndarray=_ndarray,
        affine_range=range,
        sequential_range=range,
        float32=np.float32,
        shared_hbm=_HBM,
        sbuf=_SBUF,
        psum=_PSUM,
        tanh=TANH,
        exp=EXP,
        add=ADD,
        multiply=MULTIPLY,
        subtract=SUBTRACT,
        maximum=MAXIMUM,
        max=MAX,
        square=SQUARE,
        rsqrt=RSQRT,
        copy=_nl_copy,
    )


def _build_nisa() -> SimpleNamespace:
    """Build mock nki.isa namespace with all ISA operations."""
    return SimpleNamespace(
        dma_copy=_dma_copy,
        nc_matmul=_nc_matmul,
        nc_transpose=_nc_transpose,
        tensor_copy=_tensor_copy,
        activation=_activation,
        tensor_tensor=_tensor_tensor,
        tensor_scalar=_tensor_scalar,
        tensor_reduce=_tensor_reduce,
    )


def _build_globals() -> dict[str, Any]:
    """Build exec globals with mock NKI modules."""
    nl_ns = _build_nl()
    nisa_ns = _build_nisa()
    nki_ns = SimpleNamespace(jit=_nki_jit, language=nl_ns, isa=nisa_ns)
    return {"nki": nki_ns, "nl": nl_ns, "nisa": nisa_ns, "np": np, "__builtins__": __builtins__}


def _is_nki_import(line: str) -> bool:
    """Check whether a source line is an NKI import statement."""
    stripped = line.strip()
    return stripped.startswith("import nki") or stripped.startswith("from nki")


def _strip_nki_imports(source: str) -> str:
    """Remove NKI import lines so exec uses injected mocks."""
    lines = source.splitlines()
    kept = [ln for ln in lines if not _is_nki_import(ln)]
    return "\n".join(kept)


def simulate_kernel(nki_source: str, func_name: str, kernel_kwargs: dict[str, np.ndarray]) -> np.ndarray:
    """Execute an NKI kernel on CPU using numpy at float64 precision.

    Replaces NKI modules with numpy-backed mocks and executes the
    kernel function to produce output without Neuron hardware.

    Args:
        nki_source: Complete NKI kernel source code string.
        func_name: Name of the @nki.jit kernel function.
        kernel_kwargs: Input arrays keyed by parameter name.

    Returns:
        Output numpy array at float64 precision.
    """
    source = _strip_nki_imports(nki_source)
    g = _build_globals()
    exec(source, g)  # noqa: S102
    kernel_fn = g[func_name]
    f64_kwargs = {k: v.astype(np.float64) for k, v in kernel_kwargs.items()}
    return kernel_fn(**f64_kwargs)
