"""Tensor copy NKI statement."""

from dataclasses import dataclass
from typing import Any

from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import NKIOp, _render_ref


@dataclass(frozen=True)
class NKITensorCopy(NKIOp):
    """Tensor copy: ``nisa.tensor_copy(dst=ref, src=ref)``.

    Attributes:
        dst: Destination tensor reference.
        src: Source tensor reference.
    """

    dst: TensorRef
    src: TensorRef

    def simulate(self, env: dict[str, Any]) -> None:
        """Copy slice data from src to dst.

        Args:
            env: Mutable variable environment.
        """
        src_data = env[self.src.name][self.src.to_slices()]
        env[self.dst.name][self.dst.to_slices()] = src_data

    def render(self) -> str:
        """Render as NKI tensor copy statement.

        Returns:
            NKI source line.
        """
        return f"nisa.tensor_copy(dst={_render_ref(self.dst)}, src={_render_ref(self.src)})"
