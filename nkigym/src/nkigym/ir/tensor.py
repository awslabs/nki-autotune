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

    def to_slices(self) -> tuple[slice, ...]:
        """Convert (start, stop) pairs to a tuple of slice objects.

        Returns:
            Tuple of slice objects for numpy indexing.
        """
        return tuple(slice(s, e) for s, e in self.slices)


def full_slices(shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Build full-range slices from a shape.

    Args:
        shape: Tensor shape tuple.

    Returns:
        Per-axis (0, size) bounds.
    """
    return tuple((0, s) for s in shape)


def ref_name(ref: Any) -> str:
    """Extract variable name from a TensorRef or plain string.

    Args:
        ref: A TensorRef or string.

    Returns:
        The variable name.
    """
    result = ref.name if isinstance(ref, TensorRef) else ref
    return result
