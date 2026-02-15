"""Typed tensor reference for GymProgram IR."""

from typing import Any, NamedTuple


class TensorRef(NamedTuple):
    """A typed reference to a tensor variable with shape and slice bounds.

    Attributes:
        name: Variable name.
        shape: Tensor shape.
        slices: Per-axis (start, stop) access bounds.
    """

    name: str
    shape: tuple[int, ...]
    slices: tuple[tuple[int, int], ...]


def ref_name(ref: Any) -> str:
    """Extract variable name from a TensorRef or plain string.

    Args:
        ref: A TensorRef or string.

    Returns:
        The variable name.
    """
    if isinstance(ref, TensorRef):
        return ref.name
    return ref
