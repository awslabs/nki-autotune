"""Lowering context and helpers shared across codegen and op modules.

This is a leaf module that only imports from ``nkigym.ir``, so it can
be imported by both ``codegen.gym_to_nki`` and the ``ops.*`` subclasses
without creating circular dependencies.
"""

from dataclasses import dataclass, field

from nkigym.ir.tensor import TensorRef
from nkigym.ir.types import GymStatement


@dataclass
class _LoweringContext:
    """Mutable state during GymProgram-to-NKI lowering.

    Attributes:
        params: Input parameter names (all live in HBM).
        buffers: Variable name to buffer location string.
        aliases: Maps accumulation output names to canonical PSUM variable.
        staging_counter: Monotonic counter for staging variable names.
    """

    params: tuple[str, ...]
    buffers: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    staging_counter: int = 0

    def resolve(self, name: str) -> str:
        """Resolve a variable name through the alias chain.

        Args:
            name: Variable name, possibly an accumulation alias.

        Returns:
            Canonical variable name.
        """
        while name in self.aliases:
            name = self.aliases[name]
        return name

    def buffer_of(self, name: str) -> str:
        """Look up the buffer location of a variable, resolving aliases.

        Args:
            name: Variable name.

        Returns:
            Buffer location string.

        Raises:
            KeyError: If variable is not tracked.
        """
        return self.buffers[self.resolve(name)]

    def subscript(self, ref: TensorRef) -> str:
        """Render a TensorRef as ``name[s:e, s:e]``, resolving aliases.

        Unconditionally renders slices from the IR. The IR is the
        source of truth — no shape comparison or optimization.

        Args:
            ref: Tensor reference.

        Returns:
            Subscripted string or plain resolved name.
        """
        resolved = self.resolve(ref.name)
        result = resolved
        if ref.slices:
            parts = ", ".join(f"{s}:{e}" for s, e in ref.slices)
            result = f"{resolved}[{parts}]"
        return result


def get_kwarg(stmt: GymStatement, key: str) -> object:
    """Extract a keyword argument value from a statement.

    Args:
        stmt: GymStatement to search.
        key: Keyword argument name.

    Returns:
        The value if found, None otherwise.
    """
    result = None
    for k, v in stmt.kwargs:
        if k == key:
            result = v
    return result
