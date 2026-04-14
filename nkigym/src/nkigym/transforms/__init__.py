"""Transform base class and KernelIR rendering.

Transform subclasses produce modified KernelIR variants.
``render_ir`` converts any KernelIR (base or transformed) to
NKI kernel source.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from nkigym.codegen.ir import KernelIR
from nkigym.codegen.render import render_ir

__all__ = ["Transform", "render_ir"]


class Transform(ABC):
    """Base class for programmatic kernel transforms.

    Each subclass implements ``candidates(ir)`` returning every
    possible single-step application of that transform.
    """

    NAME: ClassVar[str] = ""

    @abstractmethod
    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        """Return all single-step applications of this transform.

        Args:
            ir: Input kernel IR.

        Returns:
            List of new KernelIR variants with the transform applied.
        """
