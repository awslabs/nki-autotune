"""NKITransform ABC and option types for NKIKernel transforms.

All transforms follow the analyze-then-apply pattern on NKIKernel:

1. ``analyze(kernel)`` — inspect an NKIKernel and return a list of TransformOptions.
2. ``apply(kernel, option)`` — apply a single option, returning a new NKIKernel.
"""

from abc import ABC, abstractmethod
from typing import NamedTuple

from nkigym.codegen.types import NKIKernel


class StmtRef(NamedTuple):
    """Reference to a specific statement within a block.

    Attributes:
        block_name: Name of the block (e.g. ``"_block_0"``).
        stmt_idx: Index of the statement within the block body.
    """

    block_name: str
    stmt_idx: int


class TransformOption(NamedTuple):
    """A single transform opportunity identified by analyze().

    Attributes:
        ref_a: First statement reference.
        ref_b: Second statement reference.
    """

    ref_a: StmtRef
    ref_b: StmtRef


class NKITransform(ABC):
    """Base class for NKIKernel transforms.

    Subclasses implement ``analyze`` and ``apply`` which operate
    on NKIKernel directly.

    Attributes:
        name: Human-readable name for logging and diagnostics.
    """

    name: str

    @abstractmethod
    def analyze(self, kernel: NKIKernel) -> list[TransformOption]:
        """Find optimization opportunities on an NKIKernel.

        Args:
            kernel: An NKI kernel.

        Returns:
            List of TransformOption instances.
        """

    @abstractmethod
    def apply(self, kernel: NKIKernel, option: TransformOption) -> NKIKernel:
        """Apply one opportunity, return new NKIKernel.

        Args:
            kernel: NKIKernel to transform.
            option: A single option from ``analyze()``.

        Returns:
            New NKIKernel with the optimization applied.
        """


Transform = NKITransform
