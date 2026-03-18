"""Buffer allocation NKI statement."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from nkigym.ops.base import NKIOp


@dataclass(frozen=True)
class NKIAlloc(NKIOp):
    """Buffer allocation: ``dst = nl.ndarray(shape, dtype=dtype, buffer=nl.buf)``.

    Attributes:
        dst: Destination variable name.
        shape: Tensor shape tuple.
        dtype: NKI dtype string (e.g. ``"nl.float32"``).
        buffer: Buffer location (e.g. ``"psum"``, ``"sbuf"``, ``"shared_hbm"``).
    """

    dst: str
    shape: tuple[int, ...]
    dtype: str
    buffer: str

    def dst_name(self) -> str:
        """Return the destination variable name.

        Returns:
            Destination name string.
        """
        return self.dst

    def tensor_names(self) -> tuple[str, ...]:
        """Return all tensor names.

        Returns:
            Tuple containing the destination name.
        """
        return (self.dst,)

    def input_names(self) -> tuple[str, ...]:
        """Return input tensor names (none for alloc).

        Returns:
            Empty tuple.
        """
        return ()

    def renamed(self, rename_map: dict[str, str]) -> "NKIAlloc":
        """Return a copy with dst renamed, or self if unchanged.

        Args:
            rename_map: Mapping from old names to new names.

        Returns:
            Renamed NKIAlloc or self.
        """
        new_dst = rename_map.get(self.dst, self.dst)
        result: NKIAlloc = (
            self
            if new_dst == self.dst
            else NKIAlloc(dst=new_dst, shape=self.shape, dtype=self.dtype, buffer=self.buffer)
        )
        return result

    def simulate(self, env: dict[str, Any]) -> None:
        """Allocate a zero-initialized buffer in the simulation environment.

        Args:
            env: Mutable variable environment.
        """
        env[self.dst] = np.zeros(self.shape)

    def render(self) -> str:
        """Render as NKI allocation statement.

        Returns:
            NKI source line.
        """
        return f"{self.dst} = nl.ndarray({self.shape}, dtype={self.dtype}, buffer=nl.{self.buffer})"
