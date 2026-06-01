"""SBUF/PSUM → SBUF copy op: maps to ``nisa.tensor_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKITensorCopy(NKIOp):
    """Copy ``src`` into ``dst`` element-wise.

    operands:
        src: source tensor (P, F).
        dst: destination tensor (P, F) — same shape/dtype.
    """

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """``src`` must be SBUF- or PSUM-resident (drain pattern allows PSUM src)."""
        role = _operand_role(kwargs["src"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKITensorCopy(src=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return a copy of ``src``."""
        return np.array(kwargs["src"])
