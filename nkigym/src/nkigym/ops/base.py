"""NKIOp base class for math-level kernel descriptions.

Each ``NKIOp`` subclass maps 1:1 to a real ``nisa.*`` ISA instruction.
Subclasses implement ``__call__`` for CPU simulation (numpy) and declare
axis semantics and hardware limits via class attributes.
"""

import functools
from abc import abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, ClassVar

import numpy as np

"""Role lattice enforced by ``NKIOp.__call__`` on every CPU-sim call:

- ``"param"``  — HBM kernel input. Only ``NKILoad`` may consume it.
- ``"sbuf"``   — SBUF-resident tensor. Compute ops and ``NKIStore`` may
  consume it.
- ``"stored"`` — HBM output of ``NKIStore``. Only the kernel ``return``
  may consume it.

Bare ``np.ndarray`` operands (the typical entry path: a kernel called
with fresh numpy inputs) are treated as ``"param"``. Every call
produces a ``_RoleArray`` carrying the output role so downstream ops
see the correct lineage."""
_ALLOWED_INPUT_ROLES: dict[str, frozenset[str]] = {"NKILoad": frozenset({"param"}), "NKIStore": frozenset({"sbuf"})}
_DEFAULT_ALLOWED_ROLES = frozenset({"sbuf"})
_OUTPUT_ROLES: dict[str, str] = {"NKILoad": "sbuf", "NKIStore": "stored"}
_DEFAULT_OUTPUT_ROLE = "sbuf"


class AxisRole(str, Enum):
    """Per-op classification of how a loop axis carries state across iterations.

    PARALLEL iterations are independent and safe to fuse with another
    op's PARALLEL loop on the same dim. SEQUENTIAL iterations carry
    non-associative state (prefix scan, running state) and must not
    fuse with a PARALLEL loop on the same dim. ACCUMULATION iterations
    contribute to an associative reducer (sum, max); the accumulator
    is live across iterations, so fusion with another nest's PARALLEL
    loop on the same dim is illegal.
    """

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ACCUMULATION = "accumulation"


class _RoleArray(np.ndarray):
    """``np.ndarray`` subclass carrying a ``role`` tag.

    Produced by every ``NKIOp.__call__``. Subclassing ``ndarray``
    (rather than wrapping it) keeps downstream consumers that rely on
    numpy semantics — ``assert_allclose``, ``astype``, arithmetic —
    working unchanged.
    """

    def __new__(cls, array: np.ndarray, role: str) -> "_RoleArray":
        obj = np.asarray(array).view(cls)
        obj.role = role
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.role = getattr(obj, "role", _DEFAULT_OUTPUT_ROLE)


def _operand_role(value: Any) -> str | None:
    """Return the role of ``value`` if it is a tensor operand, else ``None``."""
    if isinstance(value, _RoleArray):
        return value.role
    if isinstance(value, np.ndarray):
        return "param"
    return None


def _tag_as_param(value: Any) -> Any:
    """Wrap bare ``np.ndarray`` values as ``role='param'`` ``_RoleArray``; pass through otherwise."""
    if isinstance(value, np.ndarray) and not isinstance(value, _RoleArray):
        return _RoleArray(value, "param")
    return value


class NKIOp:
    """Base for all NKI operator definitions.

    Supports two call-site syntaxes:

    * ``Op()(data=x)`` — single call, all kwargs on ``__call__``.
    * ``Op(op='square', ...)(data=x)`` — split: configuration literals
      on the constructor, tensor operands on the invocation. Constructor
      kwargs are stashed and merged into the final ``__call__`` kwargs
      at CPU-sim time.

    ``__call__`` enforces a load / compute / store lineage at CPU-sim
    time: HBM kernel parameters may only be consumed by ``NKILoad``;
    compute ops and ``NKIStore`` require SBUF-resident tensors; the
    kernel must ``return`` an ``NKIStore`` output. Violations raise
    ``TypeError`` from the offending call site — the traceback points
    directly at the bad op.

    Attributes:
        NAME: ISA call name (e.g. ``"nc_matmul"``).
        OPERAND_AXES: Maps operand name to axis label tuple.
        OUTPUT_AXES: Maps output name to axis label tuple.
        OUTPUT_DTYPES: Optional per-output dtype override. Default empty
            means outputs inherit the first operand's dtype (typical for
            elementwise/math ops). ``NKIActivationReduce`` overrides its
            reduce output to ``float32`` so the Scalar Engine can
            accumulate without narrowing.
        AXIS_ROLES: Per-op axis → role classification. Omitted axes default
            to ``AxisRole.PARALLEL``.
        TILE_LIMITS: Hardware tile size per abstract axis.
        OP_LOCAL_BUFFERS: Buffers allocated at function top but sized at
            single-iteration scope (see ClassVar docstring). Empty for
            most ops; used by reducers that need cross-phase scratch
            within one tile's iteration.
    """

    NAME: ClassVar[str] = ""
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_DTYPES: ClassVar[dict[str, str]] = {}
    AXIS_ROLES: ClassVar[dict[str, "AxisRole"]] = {}
    TILE_LIMITS: ClassVar[dict[str, int]] = {}
    OP_LOCAL_BUFFERS: ClassVar[dict[str, tuple[str, str, tuple[str, ...]]]] = {}
    """Op-local buffers allocated at function top but sized at single-iteration
    scope (no outer-dim blow-up). Each entry maps a logical name to
    ``(location, dtype, axis_ids)`` — ``location`` ∈ ``{"sbuf", "psum"}``,
    ``dtype`` is an ``nl.*`` dtype name (e.g. ``"float32"``), ``axis_ids``
    are abstract axis labels from ``OPERAND_AXES`` keys plus any op-local
    derived axes declared on the op. The emitted buffer identifiers are
    ``{location}_local_<id>`` where ``<id>`` is assigned per op instance
    monotonically across the kernel."""

    def __init__(self, **kwargs: Any) -> None:
        """Stash constructor kwargs for merging into ``__call__`` kwargs."""
        self._init_kwargs: dict[str, Any] = kwargs

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Subclass-specific numpy simulation. Gets the merged kwargs."""

    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation — check operand roles, dispatch to ``_run``, tag output."""
        merged = {**getattr(self, "_init_kwargs", {}), **kwargs}
        op_name = type(self).__name__
        allowed = _ALLOWED_INPUT_ROLES.get(op_name, _DEFAULT_ALLOWED_ROLES)
        for arg_name, value in merged.items():
            role = _operand_role(value)
            if role is None:
                continue
            if role not in allowed:
                roles_str = "|".join(sorted(allowed))
                raise TypeError(
                    f"{op_name}({arg_name}=<role={role}>) expected role in {{{roles_str}}}; "
                    f"did you forget to load or store?"
                )
        result = self._run(**merged)
        output_role = _OUTPUT_ROLES.get(op_name, _DEFAULT_OUTPUT_ROLE)
        if isinstance(result, np.ndarray):
            return _RoleArray(result, output_role)
        return result


def nkigym_kernel(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark ``func`` as an nkigym kernel and enforce load / store discipline.

    Tags every ``np.ndarray`` argument with ``role="param"`` on entry,
    so any non-``NKILoad`` op that touches it raises ``TypeError`` from
    the offending op's call site. After ``func`` returns, asserts the
    return value is a ``_RoleArray`` with ``role="stored"`` — i.e. the
    last op was ``NKIStore`` — otherwise raises ``TypeError`` at the
    return site.

    Preserves the wrapped function's signature and source; downstream
    consumers that rely on ``inspect.signature`` / ``inspect.getsource``
    (``build_ir``, the synthesis prompt builder, etc.) keep working.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tagged_args = tuple(_tag_as_param(a) for a in args)
        tagged_kwargs = {k: _tag_as_param(v) for k, v in kwargs.items()}
        result = func(*tagged_args, **tagged_kwargs)
        role = _operand_role(result)
        if role != "stored":
            raise TypeError(f"{func.__name__} returned role={role!r}; kernel must end with NKIStore before `return`")
        return result

    return wrapper
