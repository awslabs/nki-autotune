"""Fill one single-partition SBUF row."""

from collections.abc import Mapping
from typing import Any, ClassVar

from nkigym.ops.base import InitializerContract, NKIOp, _operand_role


class NKIRowMemset(NKIOp):
    """Fill ``dst(1, M)`` with one compile-time value."""

    NAME: ClassVar[str] = "memset"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"dst": ("K", "M")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"K": 1}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"K": 1, "M": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"K": 1, "M": None}

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> InitializerContract:
        """Return the configured fill contract."""
        return InitializerContract(output_operand="dst", value=float(kwargs["value"]))

    def _output_role(self, **kwargs: Any) -> str:
        """Preserve the destination residency."""
        return _operand_role(kwargs["dst"]) or "sbuf"

    def _run(self, **kwargs: Any) -> Any:
        """Fill the destination in place."""
        kwargs["dst"][...] = kwargs["value"]
        return kwargs["dst"]


__all__ = ["NKIRowMemset"]
