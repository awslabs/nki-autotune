"""Element-wise activation: dst = op(data)."""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import NKIOp, _render_ref

_NL_ACT_MAP: dict[str, str] = {"nl.tanh": "tanh", "nl.exp": "exp", "nl.identity": ""}


@dataclass(frozen=True)
class NKIActivation(NKIOp):
    """Activation: ``nisa.activation(dst=ref, data=ref, op=op)``.

    Attributes:
        dst: Destination tensor reference.
        src: Source tensor reference.
        op: Activation function string (e.g. ``"nl.tanh"``).
    """

    dst: TensorRef
    src: TensorRef
    op: str

    op_name: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[tuple[str, str]] = ("P", "F")
    TILE_LIMITS: ClassVar[dict[str, int]] = {}

    def simulate(self, env: dict[str, Any]) -> None:
        """Apply element-wise activation in the simulation environment.

        Args:
            env: Mutable variable environment.
        """
        src_data = env[self.src.name][self.src.to_slices()]
        np_name = _NL_ACT_MAP.get(self.op, "")
        activated = getattr(np, np_name)(src_data) if np_name else src_data
        env[self.dst.name][self.dst.to_slices()] = activated

    def render(self) -> str:
        """Render as NKI activation statement.

        Returns:
            NKI source line.
        """
        return f"nisa.activation(dst={_render_ref(self.dst)}, data={_render_ref(self.src)}, op={self.op})"
