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
    """

    NAME: ClassVar[str] = ""
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation using numpy at float64 precision."""

    @classmethod
    def format_isa_call(cls, dst_expr: str, operand_exprs: dict[str, str]) -> str:
        """Format the nisa.* ISA call string from dst and operand expressions."""
        raise NotImplementedError(f"{cls.NAME} must implement format_isa_call")
