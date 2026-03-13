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
