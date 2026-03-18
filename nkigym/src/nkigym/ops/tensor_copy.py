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

    def renamed(self, rename_map: dict[str, str]) -> "NKITensorCopy":
        """Return a copy with refs renamed, or self if unchanged.

        Args:
            rename_map: Mapping from old names to new names.

        Returns:
            Renamed NKITensorCopy or self.
        """
        new_dst = self.dst.renamed(rename_map)
        new_src = self.src.renamed(rename_map)
        changed = new_dst is not self.dst or new_src is not self.src
        result: NKITensorCopy = NKITensorCopy(dst=new_dst, src=new_src) if changed else self
        return result

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
