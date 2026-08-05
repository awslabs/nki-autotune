"""NKIOp base class for math-level kernel descriptions.

Each ``NKIOp`` subclass maps 1:1 to a real ``nisa.*`` ISA instruction.
Subclasses implement ``__call__`` for CPU simulation (numpy) and declare
axis semantics and hardware limits via class attributes.
"""

import functools
from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal

import numpy as np

"""Role lattice for CPU-sim role tracking:

- ``"param"``       — HBM kernel input. Only ``NKILoad`` may consume it.
- ``"sbuf"``        — SBUF-resident tensor.
- ``"psum"``        — PSUM-resident tensor.
- ``"shared_hbm"``  — non-output HBM tensor (intra-kernel scratch / final output).
- ``"stored"`` — HBM output of ``NKIStore``. Also acceptable as the kernel
  ``return`` value (alongside ``"shared_hbm"``).

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


@dataclass(frozen=True)
class ReduceCombinator:
    """An op's commutative-associative reducer, mirroring TVM's CommReducer.

    Attributes:
        combiner: the ``nl.*`` op name applied in the RFactor wb-block combine
            (e.g. ``"add"`` for a sum reduction).
        identity: the value RFactor memsets both the rf-block slot and the
            wb-block accumulator to before reducing (e.g. ``0.0`` for sum).
    """

    combiner: str
    identity: float


@dataclass(frozen=True)
class PointwiseContract:
    """Algebraic contract for an elementwise operation.

    Attributes:
        operator: Standard operation name such as ``"multiply"`` or ``"exp"``.
        input_operands: Ordered tensor-or-literal operand slots.
        output_operand: Slot receiving the result.
        broadcast_operands: Input slots broadcast over the free dimension.
        reverse: Whether a non-commutative binary operation reverses its operands.
        scale: Affine scale applied before a unary operation.
        bias: Affine bias applied before a unary operation.
        bias_operand: Optional tensor bias added before a unary operation.
    """

    operator: str
    input_operands: tuple[str, ...]
    output_operand: str
    broadcast_operands: frozenset[str] = frozenset()
    reverse: bool = False
    scale: float = 1.0
    bias: float = 0.0
    bias_operand: str | None = None


@dataclass(frozen=True)
class PointwiseSequenceContract:
    """Algebraic contract for a sequence of binary pointwise operations."""

    operators: tuple[str, ...]
    input_operands: tuple[str, ...]
    output_operand: str
    broadcast_operands: frozenset[str] = frozenset()
    reverse: tuple[bool, ...] = ()


@dataclass(frozen=True)
class ReductionContract:
    """Algebraic contract for a mapped associative reduction.

    ``map_operator`` is applied elementwise before ``combinator`` reduces
    ``reduction_axis``. ``"copy"`` denotes an identity map.
    ``bias_operand`` names an optional broadcast tensor added before the map.
    ``mapped_output_operand`` names an optional second output receiving the
    mapped tile before reduction.
    """

    input_operand: str
    output_operand: str
    reduction_axis: str
    combinator: ReduceCombinator
    map_operator: str = "copy"
    scale: float = 1.0
    bias: float = 0.0
    bias_operand: str | None = None
    mapped_output_operand: str | None = None


@dataclass(frozen=True)
class BilinearReductionContract:
    """Algebraic contract for a bilinear operation with one reduction axis."""

    left_operand: str
    right_operand: str
    output_operand: str
    reduction_axis: str
    combinator: ReduceCombinator


@dataclass(frozen=True)
class BatchedPermutationContract:
    """Hardware-supported embedding of a permutation with one batch axis.

    Attributes:
        permutation: Permutation applied to the expanded input.
        input_axes: Expanded-input positions occupied by the logical input
            axes, in logical-axis order.
        batch_axis: Expanded-input position occupied by the preserved batch
            axis. Every remaining position is a singleton axis.
    """

    permutation: tuple[int, ...]
    input_axes: tuple[int, ...]
    batch_axis: int


@dataclass(frozen=True)
class PermutationContract:
    """Algebraic contract for an axis permutation."""

    input_operand: str
    output_operand: str
    permutation: tuple[int, ...]
    batching: BatchedPermutationContract | None = None


@dataclass(frozen=True)
class CopyContract:
    """Algebraic contract for a value-preserving copy."""

    input_operand: str
    output_operand: str


@dataclass(frozen=True)
class InitializerContract:
    """Algebraic contract for filling a destination with an identity value."""

    output_operand: str
    value: float


OperatorContract = (
    PointwiseContract
    | PointwiseSequenceContract
    | ReductionContract
    | BilinearReductionContract
    | PermutationContract
    | CopyContract
    | InitializerContract
)


def reduction_combinator(name: str) -> ReduceCombinator:
    """Resolve a supported reduction name to its algebraic properties."""
    normalized = "maximum" if name == "max" else name
    combinators = {
        "add": ReduceCombinator(combiner="add", identity=0.0),
        "maximum": ReduceCombinator(combiner="maximum", identity=float("-inf")),
        "multiply": ReduceCombinator(combiner="multiply", identity=1.0),
    }
    if normalized not in combinators:
        raise ValueError(f"unsupported reduction combinator {name!r}")
    return combinators[normalized]


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
        AXIS_ROLES: Per-op axis → role classification. Omitted axes default
            to ``AxisRole.PARALLEL``.
    """

    NAME: ClassVar[str] = ""
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    AXIS_ROLES: ClassVar[dict[str, "AxisRole"]] = {}

    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {}
    """Minimum legal innermost-tile extent per abstract axis.

    Going below this extent is a hardware- or performance-floor violation.
    Split/Fuse reject atoms that would produce a smaller innermost tile.
    Empty = no floor for any axis (legal by default).
    """

    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {}
    """Maximum legal innermost-tile extent per abstract axis.

    ``None`` means unbounded. Canonical build picks the largest legal tile
    (``MAX`` when set, full extent when unset). Split/Fuse reject atoms
    that would produce a larger innermost tile.
    Empty = no cap for any axis.
    """

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

    Both recipes emit the same rf-buffer + write-back-block shape; they differ
    only in how the rf-block's per-slot accumulate lowers:

    - ``"rmw"``: ops with a HW accumulator (matmul). Per-slot accumulate is
      HW ``+=`` into a PSUM slot, drained to the SBUF rf-buffer; the wb-block
      combine is a ``tensor_tensor``.
    - ``"slot"``: reductions with no HW accumulator. Each slot is written
      directly in SBUF; the wb-block closes with a ``tensor_reduce``.
    - ``None``: RFactor legality rejects any atom targeting this op.
    """

    REDUCE_COMBINATOR: ClassVar["ReduceCombinator | None"] = None
    """The op's commutative-associative reducer, or ``None`` if not a reduction.

    RFactor reads this to synthesize the rf-block per-slot init, the wb-block
    init (both ``memset`` to ``identity``), and the wb-block combine (an ISA op
    applying ``combiner``). Must be set on every op whose ``RFACTOR_RECIPE`` is
    not ``None``.
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

    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {}
    """Physical dtype required by specific tensor input slots.

    Tracing propagates these requirements back to the bound producer buffer.
    Literal operands are unaffected.
    """

    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {}
    """Accepted physical locations for input operands used by storage rewrites.

    An absent entry means the operation has not declared that operand safe for
    copy propagation. Runtime role checks remain the authoritative validation
    for direct DSL calls.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Stash constructor kwargs for merging into ``__call__`` kwargs."""
        self._init_kwargs: dict[str, Any] = kwargs

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> OperatorContract | None:
        """Return this operation's algebraic contract for ``kwargs``.

        Operations that have not declared algebraic behavior return ``None``.
        Contract-driven transforms reject such operations when they occur on a
        candidate path.
        """
        _ = kwargs
        return None

    @classmethod
    def first_write_overwrites(cls, operand: str, kwargs: Mapping[str, Any]) -> bool:
        """Return whether the first execution overwrites one RMW operand."""
        _ = operand, kwargs
        return False

    @classmethod
    def rmw_operands(cls, kwargs: Mapping[str, Any]) -> frozenset[str]:
        """Return read-modify-write operands for one configured operation."""
        _ = kwargs
        return cls.RMW_OPERANDS

    @classmethod
    def with_first_write_overwrite(cls, operand: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Return kwargs that force the first write to overwrite."""
        if not cls.first_write_overwrites(operand, kwargs):
            raise ValueError(f"{cls.__name__}.{operand} does not support first-write overwrite")
        return dict(kwargs)

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Subclass-specific numpy simulation. Gets the merged kwargs."""

    OUTPUT_ROLE: ClassVar[str] = _DEFAULT_OUTPUT_ROLE
    """Role tag attached to ``__call__``'s return value.

    Subclasses override to declare an op-specific output role
    (e.g. ``NKIStore.OUTPUT_ROLE = "stored"``). ``NKIMemset`` overrides
    :meth:`_output_role` directly to return its ``dst`` operand's
    existing role (memset does not change residency).
    """

    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    """Physical residency of this op's synthesized output buffer:
    ``"sbuf"`` / ``"psum"`` / ``"shared_hbm"``. Read by the trace to set
    the output ``Buffer.location``. Distinct from ``OUTPUT_ROLE`` (the
    role-lattice lineage tag); the two coincide for every op except
    ``NKIStore`` (location ``"shared_hbm"``, role ``"stored"``)."""

    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = None
    """Physical dtype override for synthesized output buffers.

    ``None`` preserves the first input's logical dtype. Ops whose hardware
    destination uses a different dtype set the concrete allocation dtype here;
    for example, ``NKIMatmul`` accumulates into fp32 PSUM.
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
        by ``NKIMemset`` to return its ``dst`` operand's existing role.
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


_VALID_RETURN_ROLES = frozenset({"stored", "shared_hbm"})


def nkigym_kernel(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark ``func`` as an nkigym kernel and enforce load / store discipline.

    Tags every ``np.ndarray`` argument with ``role="param"`` on entry
    so any non-``NKILoad`` op that touches them fails its per-op role
    check. After ``func`` returns, asserts the return value is a
    ``_RoleArray`` with ``role in {"stored", "shared_hbm"}`` — either the
    direct return of an ``NKIStore`` call (``"stored"``) or the HBM
    buffer the caller allocated and stored into (``"shared_hbm"``). Other
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

    setattr(wrapper, "__nkigym_kernel__", True)
    return wrapper
