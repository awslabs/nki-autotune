"""Shared utilities for cross-block NKIKernel transforms.

Provides block concatenation, statement reference resolution, tensor
renaming, slice adjacency checking, and kernel block replacement.
"""

import dataclasses

from nkigym.codegen.types import NKIBlock, NKIKernel, _rename_stmt
from nkigym.ir.tensor import TensorRef
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
    renamed_body = tuple(_rename_stmt(s, rename_map) for s in block_b.body) if rename_map else block_b.body
    merged_params = tuple(dict.fromkeys(list(block_a.params) + list(block_b.params)))
    return NKIBlock(name=block_a.name, params=merged_params, body=block_a.body + renamed_body)


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
    return block._replace(body=tuple(_rename_stmt(s, rename_map) for s in block.body))


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
    names: set[str] = set()
    for stmt in block.body:
        names.update(_stmt_names(stmt))
    return names


def _stmt_names(stmt: NKIOp) -> list[str]:
    """Extract variable names from a statement (alloc dst + TensorRef names).

    Args:
        stmt: An NKI statement.

    Returns:
        List of variable names.
    """
    names: list[str] = []
    for fld in dataclasses.fields(stmt):
        val = getattr(stmt, fld.name)
        if isinstance(val, TensorRef):
            names.append(val.name)
        elif fld.name == "dst" and isinstance(val, str):
            names.append(val)
    return names


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
