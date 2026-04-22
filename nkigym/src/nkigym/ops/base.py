"""NKIOp base class with __call__ simulation.

Each NKIOp subclass maps 1:1 to a real nisa.* ISA instruction.
Subclasses implement __call__() for CPU simulation (numpy at float64),
declare axis semantics and hardware limits via class attributes, and
provide format_isa_call() for the generic renderer.
"""

from abc import abstractmethod
from typing import Any, ClassVar


class NKIOp:
    """Base for all NKI operator definitions.

    Subclasses define axis semantics, hardware limits, ISA call
    format, and CPU simulation.  The generic renderer reads these
    class attributes — no op-specific logic lives in the renderer.

    Attributes:
        NAME: ISA call name (e.g. ``"nc_matmul"``).
        OPERAND_AXES: Maps operand name to axis label tuple.
        OUTPUT_AXES: Maps output name to axis label tuple.
        BLOCKING_AXES: Accumulation axes (e.g. ``{"K"}`` for matmul).
        TILE_LIMITS: Hardware tile size per abstract axis.
        ISA_LOC: Where the ISA writes output (``"psum"`` or ``"sbuf"``).
        PSUM_DTYPE: Override dtype for PSUM buffer (``None`` = input dtype).
        INPUT_LOCS: Per-operand memory requirement. Maps
            ``role -> "sbuf"`` (SBUF only) or ``"sbuf_or_psum"``.
            Used by the renderer to decide PSUM→SBUF staging.
        _NKI_OP_KWARGS: Kwarg names that represent NKI op enums
            and need ``nl.`` prefix conversion.
    """

    NAME: ClassVar[str] = ""
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {}
    FLOAT32_KWARGS: ClassVar[frozenset[str]] = frozenset()
    """Per-output combinator declaring what happens when one of the
    op's blocking axes becomes a multi-chunk loop.

    Maps ``output_role -> combinator_kwarg_name`` pointing at the
    instance-level kwarg whose value names the combinator
    (``"add"``, ``"maximum"``, ``"minimum"``, ...). The renderer
    reads the kwarg to pick an init value and an ISA combine call.

    Empty dict = no blocking-axis reduction semantics (PARALLEL
    output). Operators like ``nc_matmul`` whose hardware
    accumulation is implicit still declare this so ``DimRole``
    classification is uniform. ``None`` in place of a kwarg name
    means the combinator is fixed (e.g. matmul is always ``add``).
    """
    REDUCE_COMBINATOR: ClassVar[dict[str, str]] = {}
    _NKI_OP_KWARGS: ClassVar[frozenset[str]] = frozenset({"op", "reduce_op", "cmp_op", "op0", "op1"})

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation using numpy at float64 precision."""

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format the nisa.* ISA call string from dst, operand expressions, and scalar kwargs."""
        raise NotImplementedError(f"{cls.NAME} must implement format_isa_call")

    @classmethod
    def _format_scalar_kwargs(cls, scalar_kwargs: dict[str, str] | None, exclude: set[str]) -> str:
        """Format non-tensor kwargs as keyword arguments string.

        Args:
            scalar_kwargs: All kwargs from the op call.
            exclude: Kwarg names that are tensor operands (already in operand_exprs).

        Returns:
            Comma-separated keyword args string, or empty string.
        """
        parts: list[str] = []
        if scalar_kwargs:
            for k, v in scalar_kwargs.items():
                if k in exclude or k.startswith("__"):
                    continue
                if k in cls._NKI_OP_KWARGS and v.startswith("'") and v.endswith("'"):
                    v = f"nl.{v[1:-1]}"
                parts.append(f"{k}={v}")
        result = ""
        if parts:
            result = ", " + ", ".join(parts)
        return result

    @classmethod
    def resolve_reduce_combinator(cls, output_role: str, op_kwargs: dict[str, str]) -> str | None:
        """Return the combinator (``'add'``/``'maximum'``/``'minimum'``) for an output, or None.

        Reads ``REDUCE_COMBINATOR[output_role]`` and resolves it:
        if the value is ``"__<literal>"`` (e.g. ``"__add"``), the
        combinator is fixed at ``<literal>``; otherwise the value
        names a kwarg (e.g. ``"op"``, ``"reduce_op"``) whose
        value at call-time holds the combinator string. Missing
        or mismatching kwargs return ``None`` — treated as
        PARALLEL output.
        """
        spec = cls.REDUCE_COMBINATOR.get(output_role)
        result: str | None = None
        if spec is not None:
            if spec.startswith("__"):
                result = spec[2:]
            else:
                raw = op_kwargs.get(spec)
                if raw is not None:
                    result = raw[1:-1] if raw.startswith("'") and raw.endswith("'") else raw
        return result

    @staticmethod
    def _to_nl(value: str) -> str:
        """Convert a quoted string op name to nl.* reference.

        ``"'square'"`` → ``"nl.square"``. Passes through
        values that are already unquoted or prefixed.
        """
        result = f"nl.{value[1:-1]}" if value.startswith("'") and value.endswith("'") else value
        return result
