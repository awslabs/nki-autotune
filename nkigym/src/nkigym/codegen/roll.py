"""Utility functions for loop rolling: skeleton extraction, offset
detection, and parameterized helper rendering.

Operates on primitives (tuple[NKIOp, ...], set[str], Any) to avoid
circular imports with types.py.
"""

import re
from typing import Any

from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import NKIOp, _render_ref
from nkigym.ops.dma_copy import NKIDmaCopy

_TENSOR_RE = re.compile(r"^tensor_(\d+)$")


def _zero_hbm_slices(stmt: NKIDmaCopy, hbm: set[str]) -> NKIDmaCopy:
    """Zero the HBM-side slice starts, keeping tile sizes.

    Args:
        stmt: DMA copy statement.
        hbm: Set of HBM tensor names.

    Returns:
        Modified NKIDmaCopy with zeroed HBM offsets.
    """
    is_load = stmt.src.name in hbm
    ref = stmt.src if is_load else stmt.dst
    zeroed = tuple((0, e - s) for s, e in ref.slices)
    new_ref = TensorRef(ref.name, ref.shape, zeroed)
    dst = stmt.dst if is_load else new_ref
    src = new_ref if is_load else stmt.src
    return NKIDmaCopy(dst=dst, src=src)


def _block_skeleton(body: tuple[NKIOp, ...], hbm: set[str]) -> tuple[NKIOp, ...]:
    """Normalize body for grouping: canonical rename + zero HBM offsets.

    Args:
        body: Block body statements.
        hbm: Set of HBM tensor names (params + output).

    Returns:
        Hashable tuple for equality-based grouping.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for stmt in body:
        for name in stmt.tensor_names():
            if name not in seen and _TENSOR_RE.match(name):
                seen.add(name)
                ordered.append(name)
    rename_map: dict[str, str] = {}
    for new_idx, old_name in enumerate(ordered):
        canonical = f"tensor_{new_idx}"
        if old_name != canonical:
            rename_map[old_name] = canonical
    result: list[NKIOp] = []
    for stmt in body:
        s = stmt.renamed(rename_map) if rename_map else stmt
        if isinstance(s, NKIDmaCopy):
            s = _zero_hbm_slices(s, hbm)
        result.append(s)
    return tuple(result)


def _extract_offsets(
    blocks: Any, indices: list[int], hbm: set[str]
) -> tuple[list[tuple[int, ...]], dict[tuple[int, int], int]]:
    """Find varying HBM axes and deduplicate offset vectors.

    Args:
        blocks: Sequence of blocks (typed Any to avoid NKIBlock import).
        indices: Block indices in this group.
        hbm: Set of HBM tensor names.

    Returns:
        Tuple of (unique offset vectors, (stmt_idx, axis) to param index).
    """
    template = blocks[indices[0]]
    seen: dict[tuple[int, ...], int] = {}
    unique: list[tuple[int, ...]] = []
    mapping: dict[tuple[int, int], int] = {}
    for stmt_idx, stmt in enumerate(template.body):
        if not isinstance(stmt, NKIDmaCopy):
            continue
        field = "src" if stmt.src.name in hbm else "dst"
        ref0: TensorRef = getattr(stmt, field)
        for axis in range(len(ref0.slices)):
            offsets = tuple(getattr(blocks[i].body[stmt_idx], field).slices[axis][0] for i in indices)
            if len(set(offsets)) <= 1:
                continue
            if offsets not in seen:
                seen[offsets] = len(unique)
                unique.append(offsets)
            mapping[(stmt_idx, axis)] = seen[offsets]
    return unique, mapping


def _render_ref_expr(ref: TensorRef, axis_offsets: dict[int, str]) -> str:
    """Render TensorRef subscript with offset parameter expressions.

    Args:
        ref: Tensor reference from template block.
        axis_offsets: Mapping from axis index to offset parameter name.

    Returns:
        Rendered subscript expression like ``x[off_0:off_0 + 128, 0:128]``.
    """
    parts = []
    for axis, (s, e) in enumerate(ref.slices):
        tile = e - s
        if axis in axis_offsets:
            p = axis_offsets[axis]
            part = f"{p}:{p} + {tile}"
        else:
            part = f"{s}:{e}"
        parts.append(part)
    return f"{ref.name}[{', '.join(parts)}]"


def _rolled_helper_name(block_names: list[str]) -> str:
    """Build helper name from block names: ``_block_0``, ``_block_1`` → ``_block_0_1``.

    Uses range notation (``_block_0_to_255``) for large contiguous groups.

    Args:
        block_names: List of block names (e.g. ``["_block_0", "_block_1"]``).

    Returns:
        Merged name with sorted indices (e.g. ``"_block_0_1"``).
    """
    indices: list[int] = []
    for name in block_names:
        indices.extend(int(x) for x in name.removeprefix("_block_").split("_"))
    sorted_idx = sorted(set(indices))
    is_range = len(sorted_idx) > 8 and sorted_idx == list(range(sorted_idx[0], sorted_idx[-1] + 1))
    suffix = f"{sorted_idx[0]}_to_{sorted_idx[-1]}" if is_range else "_".join(str(i) for i in sorted_idx)
    return "_block_" + suffix


def _render_rolled_helper(
    body: tuple[NKIOp, ...],
    kernel_params: tuple[str, ...],
    varying_map: dict[tuple[int, int], int],
    param_names: list[str],
    helper_name: str,
    hbm: set[str],
) -> list[str]:
    """Render parameterized helper via render-then-replace.

    Args:
        body: Template block body statements.
        kernel_params: Kernel input parameter names.
        varying_map: Map from (stmt_idx, axis) to param index.
        param_names: Offset parameter names.
        helper_name: Function name for the helper.
        hbm: Set of HBM tensor names.

    Returns:
        List of source lines for the helper function.
    """
    sig = ", ".join(list(kernel_params) + ["output"] + param_names)
    lines = [f"def {helper_name}({sig}):"]
    for i, stmt in enumerate(body):
        has_var = any(s == i for s, _ in varying_map)
        if not isinstance(stmt, NKIDmaCopy) or not has_var:
            lines.append(f"    {stmt.render()}")
            continue
        original = stmt.render()
        field = "src" if stmt.src.name in hbm else "dst"
        ref: TensorRef = getattr(stmt, field)
        old_str = _render_ref(ref)
        axis_offsets = {
            ax: param_names[varying_map[(i, ax)]] for ax in range(len(ref.slices)) if (i, ax) in varying_map
        }
        new_str = _render_ref_expr(ref, axis_offsets)
        lines.append(f"    {original.replace(old_str, new_str)}")
    return lines
