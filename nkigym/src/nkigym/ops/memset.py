"""In-place fill op: maps to ``nisa.memset``."""

from typing import Any, ClassVar

from nkigym.ops.base import NKIOp, _operand_role


class NKIMemset(NKIOp):
    """Fill a tensor with a constant value.

    kwargs:
        value: ``float`` — the constant to write into every element.
    operands:
        dst: target tensor (P, F layout).
    """

    NAME: ClassVar[str] = "memset"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}

    def _output_role(self, **kwargs: Any) -> str:
        """Return the role of ``dst`` unchanged (memset does not change residency)."""
        role = _operand_role(kwargs["dst"])
        return role if role is not None else "sbuf"

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: in-place fill of dst with value."""
        dst = kwargs["dst"]
        value = kwargs["value"]
        dst[...] = value
        return dst
