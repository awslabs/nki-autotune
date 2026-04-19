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

    @staticmethod
    def _to_nl(value: str) -> str:
        """Convert a quoted string op name to nl.* reference.

        ``"'square'"`` → ``"nl.square"``. Passes through
        values that are already unquoted or prefixed.
        """
        result = f"nl.{value[1:-1]}" if value.startswith("'") and value.endswith("'") else value
        return result
