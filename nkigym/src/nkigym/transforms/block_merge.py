"""Shared utilities for cross-block NKIKernel transforms.

Provides block concatenation, statement reference resolution, tensor
renaming, slice adjacency checking, and kernel block replacement.
"""

from nkigym.codegen.types import NKIBlock, NKIKernel
from nkigym.ops.base import NKIOp
from nkigym.transforms.base import TransformOption


def resolve_option(kernel: NKIKernel, option: TransformOption) -> tuple[NKIKernel, NKIBlock, int, int]:
    """Dereference StmtRefs, concatenating blocks if cross-block.

    Args:
        kernel: The NKI kernel.
        option: Transform option with two StmtRefs.

    Returns:
        Tuple of (possibly-modified kernel, target block, idx_a, idx_b).
    """
    ref_a, ref_b = option.ref_a, option.ref_b
    block_a = _find_block(kernel, ref_a.block_name)
    result_kernel = kernel
    if ref_a.block_name == ref_b.block_name:
        result_block = block_a
        result_idx_a = ref_a.stmt_idx
        result_idx_b = ref_b.stmt_idx
    else:
        block_b = _find_block(kernel, ref_b.block_name)
        merged = concat_blocks(block_a, block_b)
        result_idx_a = ref_a.stmt_idx
        result_idx_b = len(block_a.body) + ref_b.stmt_idx
        result_kernel = _replace_blocks(kernel, block_a.name, block_b.name, merged)
        result_block = merged
    return (result_kernel, result_block, result_idx_a, result_idx_b)


def concat_blocks(block_a: NKIBlock, block_b: NKIBlock) -> NKIBlock:
    """Concatenate two blocks, renaming conflicting tensor names in block_b.

    Args:
        block_a: First block (names preserved).
        block_b: Second block (names remapped if conflicting).

    Returns:
        Merged NKIBlock with combined body.
    """
    names_a = _collect_tensor_names(block_a)
    names_b = _collect_tensor_names(block_b)
    conflicts = names_a & names_b
    rename_map: dict[str, str] = {}
    if conflicts:
        max_idx = _max_tensor_idx(names_a | names_b)
        for old_name in sorted(conflicts):
            max_idx += 1
            rename_map[old_name] = f"tensor_{max_idx}"
    renamed_body = tuple(s.renamed(rename_map) for s in block_b.body) if rename_map else block_b.body
    merged_params = tuple(dict.fromkeys(list(block_a.params) + list(block_b.params)))
    name = _merged_block_name(block_a.name, block_b.name)
    return NKIBlock(name=name, params=merged_params, body=block_a.body + renamed_body)


def rename_refs(block: NKIBlock, old: str, new: str) -> NKIBlock:
    """Rename all occurrences of a tensor name in a block.

    Args:
        block: The block to modify.
        old: Old tensor name.
        new: New tensor name.

    Returns:
        New block with renamed references.
    """
    rename_map = {old: new}
    return block._replace(body=tuple(s.renamed(rename_map) for s in block.body))


def check_adjacent(
    slices_a: tuple[tuple[int, int], ...], slices_b: tuple[tuple[int, int], ...]
) -> tuple[int, tuple[int, int]]:
    """Find a single differing dimension with adjacent ranges.

    Args:
        slices_a: Slice tuple from the first operand.
        slices_b: Slice tuple from the second operand.

    Returns:
        Tuple of (dim, merged_range) if adjacent, or (-1, (0, 0)) if not.
    """
    result: tuple[int, tuple[int, int]] = (-1, (0, 0))
    diffs = [d for d in range(len(slices_a)) if slices_a[d] != slices_b[d]]
    if len(slices_a) == len(slices_b) and len(diffs) == 1:
        dim = diffs[0]
        sa_start, sa_stop = slices_a[dim]
        sb_start, sb_stop = slices_b[dim]
        if sa_stop == sb_start:
            result = (dim, (sa_start, sb_stop))
        elif sb_stop == sa_start:
            result = (dim, (sb_start, sa_stop))
    return result


def find_adjacent_pairs(
    items: list[tuple[int, tuple[tuple[int, int], ...]]], limits: tuple[int, ...]
) -> list[tuple[int, int, int, tuple[int, int]]]:
    """Find all adjacent (idx_a, idx_b, dim, merged_range) via end-to-start matching.

    For each dimension d, groups items by all other dimensions, then matches
    items whose end on dim d equals another item's start. O(n * ndim) instead
    of O(n^2).

    Args:
        items: List of (index, slices) pairs.
        limits: Per-dimension maximum merged size.

    Returns:
        List of (idx_a, idx_b, dim, merged_range) tuples.
    """
    results: list[tuple[int, int, int, tuple[int, int]]] = []
    ndim = len(items[0][1]) if items else 0
    for d in range(ndim):
        _adjacent_on_dim(items, d, limits, results)
    return results


def _adjacent_on_dim(
    items: list[tuple[int, tuple[tuple[int, int], ...]]],
    dim: int,
    limits: tuple[int, ...],
    results: list[tuple[int, int, int, tuple[int, int]]],
) -> None:
    """Find adjacent pairs on a single dimension via end-to-start matching.

    Groups items by non-target dimensions, then within each subgroup
    matches items whose end on dim equals another's start. Inlined
    for performance (eliminates 4.5M function calls in the search).

    Args:
        items: List of (index, slices) pairs.
        dim: Dimension to check adjacency on.
        limits: Per-dimension maximum merged size.
        results: Mutable list to append results to.
    """
    subgroups: dict[tuple[tuple[int, int], ...], list[tuple[int, tuple[tuple[int, int], ...]]]] = {}
    for idx, slices in items:
        other = slices[:dim] + slices[dim + 1 :]
        subgroups.setdefault(other, []).append((idx, slices))
    limit = limits[dim]
    for sub in subgroups.values():
        by_end: dict[int, list[tuple[int, tuple[tuple[int, int], ...]]]] = {}
        for idx, slices in sub:
            by_end.setdefault(slices[dim][1], []).append((idx, slices))
        for idx, slices in sub:
            for m_idx, m_slices in by_end.get(slices[dim][0], []):
                merged_start = m_slices[dim][0]
                merged_end = slices[dim][1]
                if merged_end - merged_start <= limit:
                    results.append((m_idx, idx, dim, (merged_start, merged_end)))


def widen_slice(
    slices: tuple[tuple[int, int], ...], dim: int, new_range: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    """Replace one dimension's bounds in a slice tuple.

    Args:
        slices: Original slice tuple.
        dim: Dimension to widen.
        new_range: New (start, stop) for that dimension.

    Returns:
        New slice tuple with the widened dimension.
    """
    return (*slices[:dim], new_range, *slices[dim + 1 :])


def replace_block(kernel: NKIKernel, name: str, new_block: NKIBlock) -> NKIKernel:
    """Swap one block in a kernel by name.

    Args:
        kernel: The NKI kernel.
        name: Name of the block to replace.
        new_block: Replacement block.

    Returns:
        New kernel with the block replaced.
    """
    blocks = tuple(new_block if b.name == name else b for b in kernel.blocks)
    return kernel._replace(blocks=blocks)


def remove_block(kernel: NKIKernel, name: str) -> NKIKernel:
    """Drop a block from the kernel by name.

    Args:
        kernel: The NKI kernel.
        name: Name of the block to remove.

    Returns:
        New kernel without the named block.
    """
    blocks = tuple(b for b in kernel.blocks if b.name != name)
    return kernel._replace(blocks=blocks)


def _block_indices(name: str) -> list[int]:
    """Extract numeric indices from a block name like ``_block_0_1_2``.

    Args:
        name: Block name with ``_block_`` prefix.

    Returns:
        Sorted list of integer indices.
    """
    return [int(x) for x in name.removeprefix("_block_").split("_")]


def _merged_block_name(name_a: str, name_b: str) -> str:
    """Combine two block names into a sorted merged name.

    Args:
        name_a: First block name (e.g. ``"_block_0"``).
        name_b: Second block name (e.g. ``"_block_1"``).

    Returns:
        Merged name with sorted indices (e.g. ``"_block_0_1"``).
    """
    indices = sorted(set(_block_indices(name_a) + _block_indices(name_b)))
    return "_block_" + "_".join(str(i) for i in indices)


def _find_block(kernel: NKIKernel, name: str) -> NKIBlock:
    """Look up a block by name.

    Args:
        kernel: The NKI kernel.
        name: Block name to find.

    Returns:
        The matching NKIBlock.

    Raises:
        KeyError: If no block with that name exists.
    """
    for block in kernel.blocks:
        if block.name == name:
            return block
    raise KeyError(f"Block {name!r} not found")


def _collect_tensor_names(block: NKIBlock) -> set[str]:
    """Collect all tensor_N variable names from a block's body.

    Args:
        block: The block to scan.

    Returns:
        Set of tensor variable names.
    """
    params = set(block.params)
    names: set[str] = set()
    for stmt in block.body:
        names.update(n for n in _stmt_names(stmt) if n not in params)
    return names


def _stmt_names(stmt: NKIOp) -> tuple[str, ...]:
    """Extract variable names from a statement via tensor_names().

    Args:
        stmt: An NKI statement.

    Returns:
        Tuple of variable names.
    """
    return stmt.tensor_names()


def _max_tensor_idx(names: set[str]) -> int:
    """Find the maximum tensor_N index from a set of names.

    Args:
        names: Set of tensor variable names.

    Returns:
        Maximum index, or -1 if no tensor_N names found.
    """
    max_idx = -1
    for name in names:
        if name.startswith("tensor_"):
            idx = int(name[7:])
            max_idx = max(max_idx, idx)
    return max_idx


def _replace_blocks(kernel: NKIKernel, name_a: str, name_b: str, merged: NKIBlock) -> NKIKernel:
    """Replace block_a with merged, remove block_b.

    Args:
        kernel: The NKI kernel.
        name_a: Name of block to replace with merged.
        name_b: Name of block to remove.
        merged: The merged block.

    Returns:
        New kernel with blocks updated.
    """
    blocks: list[NKIBlock] = []
    for b in kernel.blocks:
        if b.name == name_a:
            blocks.append(merged)
        elif b.name != name_b:
            blocks.append(b)
    return kernel._replace(blocks=tuple(blocks))
