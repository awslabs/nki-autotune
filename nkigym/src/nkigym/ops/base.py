"""NKIOp base class for math-level kernel descriptions.

Each ``NKIOp`` subclass maps 1:1 to a real ``nisa.*`` ISA instruction.
Subclasses implement ``__call__`` for CPU simulation (numpy) and declare
axis semantics and hardware limits via class attributes.
"""

from abc import abstractmethod
from typing import Any, ClassVar


class NKIOp:
    """Base for all NKI operator definitions.

    Attributes:
        NAME: ISA call name (e.g. ``"nc_matmul"``).
        OPERAND_AXES: Maps operand name to axis label tuple.
        OUTPUT_AXES: Maps output name to axis label tuple.
        BLOCKING_AXES: Accumulation axes (e.g. ``{"K"}`` for matmul).
        TILE_LIMITS: Hardware tile size per abstract axis.
    """

    NAME: ClassVar[str] = ""
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {}

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation using numpy."""
