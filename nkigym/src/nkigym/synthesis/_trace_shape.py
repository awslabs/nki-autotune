"""Shape and axis normalization for symbolic NumPy tracing."""

from __future__ import annotations

import numpy as np


def shape_argument(value: object) -> tuple[int, ...]:
    """Normalize a NumPy shape argument."""
    if isinstance(value, int):
        shape = (value,)
    elif isinstance(value, (tuple, list)) and all(isinstance(item, int) for item in value):
        shape = tuple(item for item in value if isinstance(item, int))
    else:
        raise ValueError(f"invalid shape argument {value!r}")
    return shape


def axes_argument(value: object) -> tuple[int, ...]:
    """Normalize one integer axis or an axis sequence."""
    if isinstance(value, int):
        axes = (value,)
    elif isinstance(value, (tuple, list)) and all(isinstance(item, int) for item in value):
        axes = tuple(item for item in value if isinstance(item, int))
    else:
        raise ValueError(f"invalid axis argument {value!r}")
    return axes


def optional_axes_argument(value: object) -> int | tuple[int, ...] | None:
    """Normalize an optional squeeze axis argument."""
    if value is None or isinstance(value, int):
        axes: int | tuple[int, ...] | None = value
    else:
        axes = axes_argument(value)
    return axes


def one_axis(axis: object, rank: int) -> int:
    """Normalize one reduction axis."""
    if isinstance(axis, tuple):
        if len(axis) != 1:
            raise ValueError(f"programmatic synthesis only supports one reduction axis, got {axis}")
        axis = axis[0]
    if not isinstance(axis, int):
        raise ValueError("programmatic synthesis requires an explicit integer reduction axis")
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"reduction axis {axis} is out of range for rank {rank}")
    return normalized


def normalize_axes(axis: int | tuple[int, ...], rank: int) -> tuple[int, ...]:
    """Normalize a tuple of unique axes."""
    raw_axes = (axis,) if isinstance(axis, int) else axis
    normalized = tuple(item + rank if item < 0 else item for item in raw_axes)
    if any(item < 0 or item >= rank for item in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError(f"invalid axes {raw_axes} for rank {rank}")
    return normalized


def normalize_transpose_axes(rank: int, axes: tuple[int | tuple[int, ...], ...]) -> tuple[int, ...]:
    """Normalize method-style transpose axes."""
    if not axes:
        normalized = tuple(reversed(range(rank)))
    elif len(axes) == 1 and isinstance(axes[0], tuple):
        normalized = normalize_axes(axes[0], rank)
    else:
        if not all(isinstance(axis, int) for axis in axes):
            raise ValueError(f"invalid transpose axes {axes}")
        normalized = normalize_axes(tuple(axis for axis in axes if isinstance(axis, int)), rank)
    return normalized


def normalize_reshape(original: tuple[int, ...], requested: tuple[int, ...]) -> tuple[int, ...]:
    """Resolve one inferred dimension and reject non-singleton reshaping."""
    if requested.count(-1) > 1 or any(dimension == 0 or dimension < -1 for dimension in requested):
        raise ValueError(f"invalid reshape target {requested}")
    resolved = requested
    if -1 in requested:
        known = int(np.prod([dimension for dimension in requested if dimension != -1]))
        total = int(np.prod(original))
        if known == 0 or total % known != 0:
            raise ValueError(f"cannot infer reshape {original} -> {requested}")
        inferred = total // known
        resolved = tuple(inferred if dimension == -1 else dimension for dimension in requested)
    if int(np.prod(original)) != int(np.prod(resolved)):
        raise ValueError(f"reshape changes element count: {original} -> {resolved}")
    if tuple(dimension for dimension in original if dimension != 1) != tuple(
        dimension for dimension in resolved if dimension != 1
    ):
        raise ValueError(f"programmatic synthesis only supports singleton reshape, got {original} -> {resolved}")
    return resolved


__all__ = [
    "axes_argument",
    "normalize_axes",
    "normalize_reshape",
    "normalize_transpose_axes",
    "one_axis",
    "optional_axes_argument",
    "shape_argument",
]
