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

    def dst_name(self) -> str:
        """Return the destination tensor name.

        Returns:
            Destination name string.
        """
        return self.dst.name

    def tensor_names(self) -> tuple[str, ...]:
        """Return all tensor names.

        Returns:
            Tuple of destination and source names.
        """
        return (self.dst.name, self.src.name)

    def input_names(self) -> tuple[str, ...]:
        """Return input tensor names.

        Returns:
            Tuple containing the source name.
        """
        return (self.src.name,)

    def renamed(self, rename_map: dict[str, str]) -> "NKIActivation":
        """Return a copy with refs renamed, or self if unchanged.

        Args:
            rename_map: Mapping from old names to new names.

        Returns:
            Renamed NKIActivation or self.
        """
        new_dst = self.dst.renamed(rename_map)
        new_src = self.src.renamed(rename_map)
        changed = new_dst is not self.dst or new_src is not self.src
        result: NKIActivation = NKIActivation(dst=new_dst, src=new_src, op=self.op) if changed else self
        return result

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
