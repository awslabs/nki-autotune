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
MINIMUM = _Op("minimum")
MAX = _Op("max")
SQUARE = _Op("square")
RSQRT = _Op("rsqrt")
RECIPROCAL = _Op("reciprocal")
GREATER_EQUAL = _Op("greater_equal")

_UNARY_FNS: dict[_Op, Any] = {
    TANH: np.tanh,
    EXP: np.exp,
    SQUARE: np.square,
    RSQRT: lambda x: 1.0 / np.sqrt(x),
    RECIPROCAL: lambda x: 1.0 / x,
}

_BINARY_FNS: dict[_Op, Any] = {
    ADD: np.add,
    MULTIPLY: np.multiply,
    SUBTRACT: np.subtract,
    MAXIMUM: np.maximum,
    MINIMUM: np.minimum,
}

_REDUCE_METHODS: dict[_Op, str] = {MAX: "max", ADD: "sum", MAXIMUM: "max", MINIMUM: "min"}

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
    """Copy data with reshape between HBM and SBUF.

    Handles multi-block partition tensors where the partition dim
    is split across num_blocks.  Layout is (par, nb..., tpb..., free...)
    so contiguous row blocks in 2D map to the num_blocks axis.

    Load:  src (256,128) -> dst (128,2,1,1,1,128): block 0=rows 0-127.
    Store: src (128,2,1,1,1,128) -> dst (256,128): reverse of load.
    """
    if src.ndim == 2 and dst.ndim > 2 and src.shape[0] > dst.shape[0]:
        n_blocks = src.shape[0] // dst.shape[0]
        blocked = src.reshape(n_blocks, dst.shape[0], -1)
        dst[:] = blocked.transpose(1, 0, 2).reshape(dst.shape)
    elif dst.ndim == 2 and src.ndim > 2 and dst.shape[0] > src.shape[0]:
        flat = src.reshape(src.shape[0], -1)
        n_blocks = dst.shape[0] // src.shape[0]
        cols = flat.shape[1] // n_blocks
        blocked = flat.reshape(src.shape[0], n_blocks, cols)
        dst[:] = blocked.transpose(1, 0, 2).reshape(dst.shape)
    else:
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
    """Apply element-wise unary activation: op(data * scale + bias).

    Supports optional bias (partition-dim vector) and scale factor.
    """
    fn = _UNARY_FNS.get(op)
    if fn is None:
        raise ValueError(f"Unknown activation: {op}")
    d2d = _flatten_2d(data)
    bias = kwargs.get("bias")
    scale = kwargs.get("scale")
    if scale is not None:
        d2d = d2d * _flatten_2d(scale) if isinstance(scale, np.ndarray) else d2d * scale
    if bias is not None:
        bias_2d = bias.reshape(d2d.shape[0], -1)
        d2d = d2d + bias_2d
    activated = fn(d2d)
    dst[:] = activated.reshape(dst.shape)


def _tensor_tensor(dst: np.ndarray, data1: np.ndarray, data2: np.ndarray, op: _Op) -> None:
    """Apply element-wise binary operation on two tiles."""
    fn = _BINARY_FNS.get(op)
    if fn is None:
        raise ValueError(f"Unknown binary op: {op}")
    dst[:] = fn(data1, data2).reshape(dst.shape)


def _tensor_scalar(dst: np.ndarray, data: np.ndarray, op0: _Op, operand0: Any, **kwargs: Any) -> None:
    """Apply element-wise op between tile and scalar/column vector.

    operand0 may be a lower-dimensional array (broadcast via padding)
    or a scalar literal.  Supports reverse mode and compound op1/operand1.
    """
    expanded = operand0
    if isinstance(operand0, np.ndarray) and data.ndim > operand0.ndim:
        pad = data.ndim - operand0.ndim
        expanded = operand0.reshape(operand0.shape + (1,) * pad)
    fn = _BINARY_FNS.get(op0)
    if fn is None:
        raise ValueError(f"Unknown scalar op: {op0}")
    reverse0 = kwargs.get("reverse0", False)
    result = fn(expanded, data) if reverse0 else fn(data, expanded)
    op1 = kwargs.get("op1")
    if op1 is not None:
        fn1 = _BINARY_FNS.get(op1)
        if fn1 is None:
            raise ValueError(f"Unknown scalar op1: {op1}")
        operand1 = kwargs["operand1"]
        expanded1 = operand1
        if isinstance(operand1, np.ndarray) and result.ndim > operand1.ndim:
            pad1 = result.ndim - operand1.ndim
            expanded1 = operand1.reshape(operand1.shape + (1,) * pad1)
        reverse1 = kwargs.get("reverse1", False)
        result = fn1(expanded1, result) if reverse1 else fn1(result, expanded1)
    dst[:] = result.reshape(dst.shape)


def _tensor_reduce(dst: np.ndarray, data: np.ndarray, op: _Op, **kwargs: object) -> None:
    """Reduce along free dimensions, keeping partition dim.

    Flattens data to (partition, free), reduces along the specified
    axis, and reshapes result to match dst. Supports negate flag.
    """
    method = _REDUCE_METHODS.get(op)
    if method is None:
        raise ValueError(f"Unknown reduce op: {op}")
    d2d = _flatten_2d(data)
    axis = kwargs.get("axis", 1)
    reduced = getattr(d2d, method)(axis=axis)
    negate = kwargs.get("negate", False)
    if negate:
        reduced = -reduced
    dst[:] = reduced.reshape(dst.shape)


def _activation_reduce(
    dst: np.ndarray, data: np.ndarray, op: _Op, reduce_op: _Op, reduce_res: np.ndarray, **kwargs: object
) -> None:
    """Activation with simultaneous reduction: op(data * scale + bias)."""
    d2d = _flatten_2d(data)
    scale = kwargs.get("scale")
    if scale is not None:
        d2d = d2d * _flatten_2d(scale) if isinstance(scale, np.ndarray) else d2d * scale
    bias = kwargs.get("bias")
    if bias is not None:
        bias_2d = bias.reshape(d2d.shape[0], -1)
        d2d = d2d + bias_2d
    fn = _UNARY_FNS.get(op)
    if fn is None:
        raise ValueError(f"Unknown activation: {op}")
    activated = fn(d2d)
    dst[:] = activated.reshape(dst.shape)
    method = _REDUCE_METHODS.get(reduce_op)
    if method is None:
        raise ValueError(f"Unknown reduce_op: {reduce_op}")
    reduced = getattr(activated, method)(axis=1)
    reduce_res[:] += reduced.reshape(reduce_res.shape)


def _memset(dst: np.ndarray, value: float, **_: object) -> None:
    """Fill destination with a constant value (mock nisa.memset)."""
    dst[:] = value


def _affine_select(
    dst: np.ndarray,
    pattern: list[list[int]],
    offset: int,
    channel_multiplier: int,
    cmp_op: _Op,
    on_true_tile: np.ndarray,
    on_false_value: float,
) -> None:
    """Position-predicated element select.

    Computes affine_value = offset + p*channel_multiplier + f*step
    per element, compares to 0, selects on_true_tile or on_false_value.
    """
    step = pattern[0][0]
    d2d = _flatten_2d(on_true_tile)
    rows, cols = d2d.shape
    p_idx = np.arange(rows)[:, np.newaxis]
    f_idx = np.arange(cols)[np.newaxis, :]
    affine_val = offset + p_idx * channel_multiplier + f_idx * step
    _CMP_FNS = {GREATER_EQUAL: np.greater_equal}
    cmp_fn = _CMP_FNS.get(cmp_op)
    if cmp_fn is None:
        raise ValueError(f"Unknown cmp_op: {cmp_op}")
    mask = cmp_fn(affine_val, 0)
    result = np.where(mask, d2d, on_false_value)
    dst[:] = result.reshape(dst.shape)


def _nc_transpose(dst: np.ndarray, **kwargs: object) -> None:
    """PE array transpose: swap partition and free dims.

    Flattens to 2D, transposes, and reshapes back to original
    multi-dim shape.  Accepts both ``src=`` and ``data=`` kwarg names.
    """
    tile = kwargs.get("src", kwargs.get("data"))
    d2d = _flatten_2d(tile)
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
        minimum=MINIMUM,
        max=MAX,
        square=SQUARE,
        rsqrt=RSQRT,
        copy=_nl_copy,
        reciprocal=RECIPROCAL,
        greater_equal=GREATER_EQUAL,
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
        affine_select=_affine_select,
        activation_reduce=_activation_reduce,
        memset=_memset,
    )


def _build_globals() -> dict[str, Any]:
    """Build exec globals with mock NKI modules."""
    nl_ns = _build_nl()
    nisa_ns = _build_nisa()
    nki_ns = SimpleNamespace(jit=_nki_jit, language=nl_ns, isa=nisa_ns)
    return {
        "nki": nki_ns,
        "nl": nl_ns,
        "nisa": nisa_ns,
        "np": np,
        "inf": np.inf,
        "Tensor": np.ndarray,
        "__builtins__": __builtins__,
    }


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
