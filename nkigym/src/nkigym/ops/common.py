"""Shared utilities for NKIOp render methods.

Buffer layout (uniform 4D):
  ``(tile_size_P, num_tiles_P, num_tiles_F, tile_size_F)``

Flat tile index per dimension:
  ``i_block * (tpb * interleave) + i_tile * interleave + i_ig``
"""


def ind(depth: int) -> str:
    """Return indentation string for the given nesting depth.

    Args:
        depth: Number of indent levels (each level = 4 spaces).

    Returns:
        Indentation string.
    """
    return " " * (4 * depth)


def linear_expr(terms: list[tuple[str, int, int]]) -> str:
    """Build a linear index expression, omitting zero-contribution terms.

    Each term is ``(variable_name, stride, trip_count)``.  Terms whose
    ``trip_count <= 1`` are omitted because the variable is always 0
    (from a ``range(1)`` loop).

    Args:
        terms: List of (variable_name, stride, trip_count) tuples.

    Returns:
        Expression string, or ``"0"`` if all terms are omitted.
    """
    parts: list[str] = []
    for var, stride, trip_count in terms:
        if trip_count <= 1:
            continue
        if stride == 1:
            parts.append(var)
        else:
            parts.append(f"{var}*{stride}")
    return "+".join(parts) if parts else "0"
