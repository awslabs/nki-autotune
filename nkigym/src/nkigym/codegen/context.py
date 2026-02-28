"""Lowering context and helpers shared across codegen and op modules.

This is a leaf module that only imports from ``nkigym.ir``, so it can
be imported by both ``codegen.gym_to_nki`` and the ``ops.*`` subclasses
without creating circular dependencies.
"""

from dataclasses import dataclass, field

import numpy as np

from nkigym.ir.tensor import TensorRef
from nkigym.ir.types import GymStatement

_NKI_UFUNC_ALIASES: dict[str, str] = {"absolute": "abs"}


def value_to_nki(value: object) -> str:
    """Convert a Python object to its NKI source representation.

    Renders numpy ufuncs as ``nl.<name>``, numpy dtypes as
    ``nl.<dtype_name>``, and other values via ``repr()``.

    Args:
        value: Python object from IR kwargs.

    Returns:
        NKI source-level string representation.
    """
    result = repr(value)
    if isinstance(value, np.ufunc):
        name = _NKI_UFUNC_ALIASES.get(value.__name__, value.__name__)
        result = f"nl.{name}"
    elif isinstance(value, np.dtype):
        result = f"nl.{value.name}"
    elif isinstance(value, type) and issubclass(value, np.generic):
        result = f"nl.{np.dtype(value).name}"
    return result


@dataclass
class _LoweringContext:
    """Mutable state during GymProgram-to-NKI lowering.

    Attributes:
        params: Input parameter names (all live in HBM).
        dtype: NKI dtype string for SBUF buffers (e.g., ``"nl.float16"``).
        buffers: Variable name to buffer location string.
        aliases: Maps accumulation output names to canonical PSUM variable.
        alias_offsets: Maps alias names to their start offsets per axis.
        tensor_counter: Monotonic counter for generated tensor variable names.
    """

    params: tuple[str, ...]
    dtype: str
    buffers: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    alias_offsets: dict[str, tuple[int, ...]] = field(default_factory=dict)
    tensor_counter: int = 0

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

    def _resolve_offsets(self, name: str) -> tuple[int, ...]:
        """Accumulate start offsets along the alias chain.

        Args:
            name: Variable name, possibly an accumulation alias.

        Returns:
            Tuple of accumulated start offsets per axis.
        """
        offsets: list[int] = []
        while name in self.aliases:
            entry_offsets = self.alias_offsets.get(name, ())
            if not offsets:
                offsets = list(entry_offsets)
            else:
                for i, o in enumerate(entry_offsets):
                    offsets[i] += o
            name = self.aliases[name]
        return tuple(offsets)

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

        When the name resolves through an alias chain, accumulates
        start offsets and composes them with the ref slices so the
        subscript points at the correct region of the canonical buffer.

        Args:
            ref: Tensor reference.

        Returns:
            Subscripted string or plain resolved name.
        """
        resolved = self.resolve(ref.name)
        offsets = self._resolve_offsets(ref.name)
        result = resolved
        if ref.slices:
            parts = _compose_slices(ref.slices, offsets)
            result = f"{resolved}[{parts}]"
        return result


def _compose_slices(slices: tuple[tuple[int, int], ...], offsets: tuple[int, ...]) -> str:
    """Compose ref slices with alias offsets into a subscript string.

    Args:
        slices: Per-axis (start, stop) bounds from the TensorRef.
        offsets: Per-axis start offsets from the alias chain.

    Returns:
        Comma-separated ``s:e`` subscript string.
    """
    parts: list[str] = []
    for i, (s, e) in enumerate(slices):
        offset = offsets[i] if i < len(offsets) else 0
        parts.append(f"{s + offset}:{e + offset}")
    return ", ".join(parts)


def get_kwarg(stmt: GymStatement, key: str) -> object:
    """Extract a keyword argument value from a statement.

    Asserts that kwargs contain no duplicate keys, since duplicates
    indicate an IR construction bug upstream.

    Args:
        stmt: GymStatement to search.
        key: Keyword argument name.

    Returns:
        The value if found, None otherwise.
    """
    _assert_no_duplicate_kwargs(stmt)
    result = None
    for k, v in stmt.kwargs:
        if k == key:
            result = v
            break
    return result


def _assert_no_duplicate_kwargs(stmt: GymStatement) -> None:
    """Assert that a statement has no duplicate keyword argument names.

    Args:
        stmt: GymStatement to check.

    Raises:
        AssertionError: If duplicate kwarg keys are found.
    """
    keys = [k for k, _ in stmt.kwargs]
    assert len(keys) == len(set(keys)), f"Duplicate kwargs in {stmt.op} stmt '{stmt.output.name}': {keys}"
