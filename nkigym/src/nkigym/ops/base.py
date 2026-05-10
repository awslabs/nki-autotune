"""NKIOp base class for math-level kernel descriptions.

Each ``NKIOp`` subclass maps 1:1 to a real ``nisa.*`` ISA instruction.
Subclasses implement ``__call__`` for CPU simulation (numpy) and declare
axis semantics and hardware limits via class attributes.
"""

import functools
from abc import abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, ClassVar, Literal

import numpy as np

"""Role lattice for CPU-sim role tracking:

- ``"param"``   — HBM kernel input. Only ``NKILoad`` may consume it.
- ``"sbuf"``    — SBUF-resident tensor.
- ``"psum"``    — PSUM-resident tensor.
- ``"hbm"``     — non-output HBM tensor (intra-kernel scratch / final output).
- ``"stored"`` — HBM output of ``NKIStore``. Also acceptable as the kernel
  ``return`` value (alongside ``"hbm"``).

Bare ``np.ndarray`` operands (the typical entry path: a kernel called
with fresh numpy inputs) are treated as ``"param"``. Per-op ``_check_roles``
methods enforce op-specific input-role constraints; the base class does
not police kwargs centrally."""
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
    """

    NAME: ClassVar[str] = ""
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_DTYPES: ClassVar[dict[str, str]] = {}
    AXIS_ROLES: ClassVar[dict[str, "AxisRole"]] = {}
    TILE_LIMITS: ClassVar[dict[str, int]] = {}

    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    """Operand slot names that this op reads AND writes (RMW semantics).

    For ``NKIMatmul``, ``dst`` is RMW — ``nisa.nc_matmul`` accumulates into its
    PSUM destination across K iterations. Every other op has disjoint
    reads and writes; this set is empty.

    Consumed by the canonical builder's ``_make_leaf`` to populate
    ``BodyLeaf.reads_writes`` (the tensor names for these slots appear in
    ``reads_writes``, not in ``reads`` or ``writes``).
    """

    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = None
    """Which RFactor recipe this op supports, or ``None`` if not rfactorable.

    - ``"rmw"``: ops with a HW accumulator (matmul). RFactor materializes
      a staging buffer, per-outer-iteration PSUM alloc, drain to SBUF slot,
      closing tensor_reduce.
    - ``"slot"``: ops whose write operand naturally indexes by reduction
      tile (activation_reduce). RFactor points successive calls at
      successive slots of a staging buffer, closes with tensor_reduce.
    - ``None``: atom legality rejects any RFactor targeting this op.
    """

    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    """Operand slots that are read-only (inputs to the computation).

    Slots in ``INPUT_OPERANDS`` land in ``BodyLeaf.reads``; slots that
    are neither in ``INPUT_OPERANDS`` nor ``RMW_OPERANDS`` (typically
    ``dst``, ``reduce_res``) land in ``BodyLeaf.writes``.

    Required for every op subclass — the canonical builder uses this set
    to split operand slots into reads / writes / reads_writes at leaf-
    construction time.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Stash constructor kwargs for merging into ``__call__`` kwargs."""
        self._init_kwargs: dict[str, Any] = kwargs

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Subclass-specific numpy simulation. Gets the merged kwargs."""

    OUTPUT_ROLE: ClassVar[str] = _DEFAULT_OUTPUT_ROLE
    """Role tag attached to ``__call__``'s return value.

    Subclasses override to declare an op-specific output role
    (e.g. ``NKIStore.OUTPUT_ROLE = "stored"``). ``NKIAlloc`` overrides
    :meth:`_output_role` directly to select based on ``location``.
    """

    def _check_roles(self, **kwargs: Any) -> None:
        """Per-op role validation. Default: no-op.

        Subclasses override to enforce input-role constraints specific
        to their semantics (e.g. ``NKILoad`` requires ``src`` to be
        ``param``; ``NKIStore`` requires ``src`` to be ``sbuf``).
        Output slots (``dst``, ``reduce_res``) are never policed here.
        """

    def _output_role(self, **kwargs: Any) -> str:
        """Return the role to tag on ``__call__``'s output array.

        Default consults the ``OUTPUT_ROLE`` class attribute. Overridden
        by ``NKIAlloc`` to pick per-call based on ``location``.
        """
        return type(self).OUTPUT_ROLE

    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation — run per-op role check, dispatch to ``_run``, tag output."""
        merged = {**getattr(self, "_init_kwargs", {}), **kwargs}
        self._check_roles(**merged)
        result = self._run(**merged)
        if isinstance(result, np.ndarray):
            return _RoleArray(result, self._output_role(**merged))
        return result


_VALID_RETURN_ROLES = frozenset({"stored", "hbm"})


def nkigym_kernel(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark ``func`` as an nkigym kernel and enforce load / store discipline.

    Tags every ``np.ndarray`` argument with ``role="param"`` on entry
    so any non-``NKILoad`` op that touches them fails its per-op role
    check. After ``func`` returns, asserts the return value is a
    ``_RoleArray`` with ``role in {"stored", "hbm"}`` — either the
    direct return of an ``NKIStore`` call (``"stored"``) or the HBM
    buffer the caller allocated and stored into (``"hbm"``). Other
    roles raise ``TypeError`` at the return site.

    The returned wrapper carries ``__nkigym_kernel__ = True`` so public
    dispatchers (``nkigym_compile``) can distinguish it from plain numpy
    callables.

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
        if role not in _VALID_RETURN_ROLES:
            raise TypeError(
                f"{func.__name__} returned role={role!r}; expected one of {sorted(_VALID_RETURN_ROLES)} "
                f"(the HBM buffer an NKIStore wrote into, or the stored-role return of NKIStore itself)"
            )
        return result

    wrapper.__nkigym_kernel__ = True
    return wrapper
