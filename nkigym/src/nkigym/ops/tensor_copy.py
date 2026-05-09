"""SBUF/PSUM → SBUF copy op: maps to ``nisa.tensor_copy``."""

from typing import Any, ClassVar

from nkigym.ops.base import NKIOp


class NKITensorCopy(NKIOp):
    """Copy ``src`` into ``dst`` element-wise.

    operands:
        src: source tensor (P, F).
        dst: destination tensor (P, F) — same shape/dtype.
    """

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: element-wise copy."""
        src = kwargs["src"]
        dst = kwargs["dst"]
        dst[...] = src
        return dst
