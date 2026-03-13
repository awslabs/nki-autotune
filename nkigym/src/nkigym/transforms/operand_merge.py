"""Operand merge transform for NKIKernel: alloc widen + compute merge.

Finds pairs of adjacent allocs or compute stmts that can be merged into
one wider operation. Alloc widen merges two buffer allocations with
adjacent DMA load sources. Compute merge combines two matmul or
activation stmts with adjacent operand slices on one axis.
"""

import dataclasses
from itertools import combinations

from nkigym.codegen.types import NKIBlock, NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.activation import NKIActivation
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import NKIOp
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.ops.matmul import NKIMatmul
from nkigym.transforms.base import NKITransform, StmtRef, TransformOption
from nkigym.transforms.block_merge import check_adjacent, replace_block, resolve_option, widen_slice

_FIELD_TO_AXES: dict[str, str] = {"src": "data"}


class OperandMergeTransform(NKITransform):
    """Merge adjacent allocs or compute stmts into wider operations."""

    name = "operand_merge"

    def analyze(self, kernel: NKIKernel) -> list[TransformOption]:
        """Find all merge opportunities across all blocks.

        Args:
            kernel: The NKI kernel to analyze.

        Returns:
            List of TransformOption pairs.
        """
        return _find_merge_pairs(kernel)

    def apply(self, kernel: NKIKernel, option: TransformOption) -> NKIKernel:
        """Apply one merge to the kernel.

        Args:
            kernel: The NKI kernel to transform.
            option: A TransformOption from ``analyze()``.

        Returns:
            New NKIKernel with the merge applied.
        """
        return _apply_merge(kernel, option)


def _find_merge_pairs(kernel: NKIKernel) -> list[TransformOption]:
    """Collect alloc widen and compute merge pairs from all blocks."""
    pairs: list[TransformOption] = []
    for block in kernel.blocks:
        pairs.extend(_find_alloc_pairs(block))
        pairs.extend(_find_compute_pairs(block))
    return pairs


def _find_alloc_pairs(block: NKIBlock) -> list[TransformOption]:
    """Find alloc pairs with adjacent DMA load sources in one block.

    Groups allocs by (dtype, buffer, dma_src_name). Checks pairs for
    adjacent HBM source slices.
    """
    alloc_info = _build_alloc_dma_map(block)
    groups: dict[tuple, list[tuple[int, NKIDmaCopy]]] = {}
    for alloc_idx, (alloc, _, dma) in alloc_info.items():
        key = (alloc.dtype, alloc.buffer, dma.src.name)
        groups.setdefault(key, []).append((alloc_idx, dma))
    pairs: list[TransformOption] = []
    for members in groups.values():
        for (i, dma_a), (j, dma_b) in combinations(members, 2):
            dim, _ = check_adjacent(dma_a.src.slices, dma_b.src.slices)
            if dim >= 0:
                pairs.append(TransformOption(StmtRef(block.name, i), StmtRef(block.name, j)))
    return pairs


def _build_alloc_dma_map(block: NKIBlock) -> dict[int, tuple[NKIAlloc, int, NKIDmaCopy]]:
    """Map alloc stmt index to (alloc, dma_idx, dma_stmt).

    Only includes allocs filled by exactly one NKIDmaCopy.
    """
    alloc_by_name: dict[str, tuple[int, NKIAlloc]] = {}
    for i, s in enumerate(block.body):
        if isinstance(s, NKIAlloc):
            alloc_by_name[s.dst] = (i, s)
    result: dict[int, tuple[NKIAlloc, int, NKIDmaCopy]] = {}
    for i, s in enumerate(block.body):
        if isinstance(s, NKIDmaCopy) and s.dst.name in alloc_by_name:
            alloc_idx, alloc = alloc_by_name[s.dst.name]
            if alloc_idx not in result:
                result[alloc_idx] = (alloc, i, s)
    return result


def _find_compute_pairs(block: NKIBlock) -> list[TransformOption]:
    """Find compute stmt pairs with adjacent operand slices."""
    pairs: list[TransformOption] = []
    matmuls = [(i, s) for i, s in enumerate(block.body) if isinstance(s, NKIMatmul)]
    pairs.extend(_matmul_pairs(block.name, matmuls))
    acts = [(i, s) for i, s in enumerate(block.body) if isinstance(s, NKIActivation)]
    pairs.extend(_activation_pairs(block.name, acts))
    return pairs


def _matmul_pairs(block_name: str, matmuls: list[tuple[int, NKIMatmul]]) -> list[TransformOption]:
    """Check matmul pairs grouped by (stationary.name, moving.name).

    Within each group, checks for single-axis operand adjacency
    within tile limits.
    """
    groups: dict[tuple, list[tuple[int, NKIMatmul]]] = {}
    for idx, stmt in matmuls:
        key = (stmt.stationary.name, stmt.moving.name)
        groups.setdefault(key, []).append((idx, stmt))
    pairs: list[TransformOption] = []
    for members in groups.values():
        for (i, a), (j, b) in combinations(members, 2):
            if _matmul_mergeable(a, b):
                pairs.append(TransformOption(StmtRef(block_name, i), StmtRef(block_name, j)))
    return pairs


def _matmul_mergeable(a: NKIMatmul, b: NKIMatmul) -> bool:
    """Check if two matmuls are mergeable on one valid axis.

    Rejects accumulation pairs (same dst) and pairs exceeding tile
    limits. Exactly one operand must be adjacent, the other identical.
    """
    accumulation = a.dst == b.dst
    stat_dim, stat_merged = check_adjacent(a.stationary.slices, b.stationary.slices)
    mov_dim, mov_merged = check_adjacent(a.moving.slices, b.moving.slices)
    mov_eq = a.moving.slices == b.moving.slices
    stat_eq = a.stationary.slices == b.stationary.slices
    result = False
    if not accumulation and stat_dim >= 0 and mov_eq:
        axis = NKIMatmul.OPERAND_AXES["stationary"][stat_dim]
        merged_size = stat_merged[1] - stat_merged[0]
        result = merged_size <= NKIMatmul.TILE_LIMITS.get(axis, 999999)
    elif not accumulation and mov_dim >= 0 and stat_eq:
        axis = NKIMatmul.OPERAND_AXES["moving"][mov_dim]
        merged_size = mov_merged[1] - mov_merged[0]
        result = merged_size <= NKIMatmul.TILE_LIMITS.get(axis, 999999)
    return result


def _activation_pairs(block_name: str, acts: list[tuple[int, NKIActivation]]) -> list[TransformOption]:
    """Check activation pairs grouped by (src.name, op)."""
    groups: dict[tuple, list[tuple[int, NKIActivation]]] = {}
    for idx, stmt in acts:
        key = (stmt.src.name, stmt.op)
        groups.setdefault(key, []).append((idx, stmt))
    pairs: list[TransformOption] = []
    for members in groups.values():
        for (i, a), (j, b) in combinations(members, 2):
            dim, _ = check_adjacent(a.src.slices, b.src.slices)
            if dim >= 0:
                pairs.append(TransformOption(StmtRef(block_name, i), StmtRef(block_name, j)))
    return pairs


def _apply_merge(kernel: NKIKernel, option: TransformOption) -> NKIKernel:
    """Dispatch merge by statement type: alloc widen or compute merge."""
    result_kernel, block, idx_a, idx_b = resolve_option(kernel, option)
    stmt_a = block.body[idx_a]
    stmt_b = block.body[idx_b]
    new_block = block
    if isinstance(stmt_a, NKIAlloc) and isinstance(stmt_b, NKIAlloc):
        new_block = _apply_alloc_widen(block, idx_a, idx_b)
    elif isinstance(stmt_a, (NKIMatmul, NKIActivation)):
        new_block = _apply_compute_merge(block, idx_a, idx_b)
    return replace_block(result_kernel, block.name, new_block)


def _apply_alloc_widen(block: NKIBlock, idx_a: int, idx_b: int) -> NKIBlock:
    """Merge two allocs and their DMA loads, remap consumers.

    Ensures the kept alloc starts before the absorbed alloc on the
    merge dimension. Widens alloc shape and DMA slices, then shifts
    absorbed tensor consumer references by the appropriate offset.
    """
    alloc_dma = _build_alloc_dma_map(block)
    alloc_a, dma_a_idx, dma_a = alloc_dma[idx_a]
    alloc_b, dma_b_idx, dma_b = alloc_dma[idx_b]
    dim, merged = check_adjacent(dma_a.src.slices, dma_b.src.slices)
    if dma_a.src.slices[dim][0] > dma_b.src.slices[dim][0]:
        idx_a, idx_b = idx_b, idx_a
        alloc_a, dma_a_idx, dma_a = alloc_dma[idx_a]
        alloc_b, dma_b_idx, dma_b = alloc_dma[idx_b]
    offset = alloc_a.shape[dim]
    new_size = alloc_a.shape[dim] + alloc_b.shape[dim]
    new_shape = (*alloc_a.shape[:dim], new_size, *alloc_a.shape[dim + 1 :])
    new_alloc = NKIAlloc(dst=alloc_a.dst, shape=new_shape, dtype=alloc_a.dtype, buffer=alloc_a.buffer)
    new_dma = _widen_dma(dma_a, new_shape, dim, merged)
    drop = {idx_b, dma_b_idx}
    replace: dict[int, NKIOp] = {idx_a: new_alloc, dma_a_idx: new_dma}
    return _rebuild_body(block, drop, replace, alloc_b.dst, alloc_a.dst, dim, offset)


def _apply_compute_merge(block: NKIBlock, idx_a: int, idx_b: int) -> NKIBlock:
    """Merge two compute stmts with adjacent operand slices.

    Widens the kept stmt operands and output. Widens the kept output
    alloc. Removes the absorbed stmt and shifts its consumer refs.
    """
    stmt_a = block.body[idx_a]
    stmt_b = block.body[idx_b]
    assert isinstance(stmt_a, (NKIMatmul, NKIActivation))
    assert isinstance(stmt_b, (NKIMatmul, NKIActivation))
    operand, op_dim, merged_range = _compute_merge_info(stmt_a, stmt_b)
    if _operand_start(stmt_a, operand, op_dim) > _operand_start(stmt_b, operand, op_dim):
        idx_a, idx_b = idx_b, idx_a
        stmt_a, stmt_b = block.body[idx_a], block.body[idx_b]
        assert isinstance(stmt_a, (NKIMatmul, NKIActivation))
        assert isinstance(stmt_b, (NKIMatmul, NKIActivation))
    out_dim = _output_dim_for_operand(type(stmt_a), operand, op_dim)
    a_size = stmt_a.dst.slices[out_dim][1] - stmt_a.dst.slices[out_dim][0]
    b_size = stmt_b.dst.slices[out_dim][1] - stmt_b.dst.slices[out_dim][0]
    out_merged = (0, a_size + b_size)
    new_stmt = _widen_compute(stmt_a, operand, op_dim, merged_range, out_dim, out_merged)
    replace: dict[int, NKIOp] = {idx_a: new_stmt}
    alloc_idx = _find_alloc_idx(block, stmt_a.dst.name)
    if alloc_idx >= 0:
        old_alloc = block.body[alloc_idx]
        assert isinstance(old_alloc, NKIAlloc)
        new_shape = (*old_alloc.shape[:out_dim], out_merged[1], *old_alloc.shape[out_dim + 1 :])
        replace[alloc_idx] = NKIAlloc(
            dst=old_alloc.dst, shape=new_shape, dtype=old_alloc.dtype, buffer=old_alloc.buffer
        )
    drop = {idx_b}
    old_name = stmt_b.dst.name
    new_name = stmt_a.dst.name
    offset = a_size if old_name != new_name else 0
    return _rebuild_body(block, drop, replace, old_name, new_name, out_dim, offset)


def _rebuild_body(
    block: NKIBlock, drop: set[int], replace: dict[int, NKIOp], old_name: str, new_name: str, dim: int, offset: int
) -> NKIBlock:
    """Rebuild block body: drop stmts, apply replacements, shift refs."""
    new_body: list[NKIOp] = []
    for i, s in enumerate(block.body):
        if i in drop:
            continue
        stmt = replace[i] if i in replace else s
        new_body.append(_shift_refs(stmt, old_name, new_name, dim, offset))
    return block._replace(body=tuple(new_body))


def _shift_refs(stmt: NKIOp, old_name: str, new_name: str, dim: int, offset: int) -> NKIOp:
    """Rename old_name to new_name and shift slices on dim by offset.

    Only shifts TensorRef fields whose name matches old_name.
    For str dst fields (NKIAlloc), just renames.
    """
    kwargs: dict[str, object] = {}
    changed = False
    for fld in dataclasses.fields(stmt):
        val = getattr(stmt, fld.name)
        if isinstance(val, TensorRef) and val.name == old_name:
            shifted = _shift_slice(val.slices, dim, offset)
            kwargs[fld.name] = TensorRef(new_name, val.shape, shifted)
            changed = True
        elif fld.name == "dst" and isinstance(val, str) and val == old_name:
            kwargs[fld.name] = new_name
            changed = True
        else:
            kwargs[fld.name] = val
    result = type(stmt)(**kwargs) if changed else stmt
    return result


def _shift_slice(slices: tuple[tuple[int, int], ...], dim: int, offset: int) -> tuple[tuple[int, int], ...]:
    """Shift one dimension's bounds by offset."""
    s, e = slices[dim]
    return (*slices[:dim], (s + offset, e + offset), *slices[dim + 1 :])


def _widen_dma(dma: NKIDmaCopy, new_shape: tuple[int, ...], dim: int, merged_src: tuple[int, int]) -> NKIDmaCopy:
    """Widen a DMA copy's dst and src slices on the merge dimension."""
    new_dst_slices = widen_slice(dma.dst.slices, dim, (0, new_shape[dim]))
    new_src_slices = widen_slice(dma.src.slices, dim, merged_src)
    return NKIDmaCopy(
        dst=TensorRef(dma.dst.name, new_shape, new_dst_slices),
        src=TensorRef(dma.src.name, dma.src.shape, new_src_slices),
    )


def _widen_ref(ref: TensorRef, dim: int, new_range: tuple[int, int]) -> TensorRef:
    """Widen a TensorRef slices and derive shape from new slices."""
    new_slices = widen_slice(ref.slices, dim, new_range)
    new_shape = tuple(e - s for s, e in new_slices)
    return TensorRef(ref.name, new_shape, new_slices)


def _widen_compute(
    stmt: NKIOp, op_field: str, op_dim: int, op_merged: tuple[int, int], out_dim: int, out_merged: tuple[int, int]
) -> NKIOp:
    """Widen a compute stmt operand and output slices."""
    kwargs: dict[str, object] = {}
    for fld in dataclasses.fields(stmt):
        val = getattr(stmt, fld.name)
        if fld.name == op_field and isinstance(val, TensorRef):
            kwargs[fld.name] = _widen_ref(val, op_dim, op_merged)
        elif fld.name == "dst" and isinstance(val, TensorRef):
            kwargs[fld.name] = _widen_ref(val, out_dim, out_merged)
        else:
            kwargs[fld.name] = val
    return type(stmt)(**kwargs)


def _compute_merge_info(stmt_a: NKIOp, stmt_b: NKIOp) -> tuple[str, int, tuple[int, int]]:
    """Determine which operand is adjacent and on which dimension."""
    result: tuple[str, int, tuple[int, int]] = ("src", -1, (0, 0))
    if isinstance(stmt_a, NKIMatmul) and isinstance(stmt_b, NKIMatmul):
        result = _matmul_merge_info(stmt_a, stmt_b)
    elif isinstance(stmt_a, NKIActivation) and isinstance(stmt_b, NKIActivation):
        result = _activation_merge_info(stmt_a, stmt_b)
    return result


def _matmul_merge_info(a: NKIMatmul, b: NKIMatmul) -> tuple[str, int, tuple[int, int]]:
    """Find the adjacent operand and merge dimension for two matmuls."""
    stat_dim, stat_merged = check_adjacent(a.stationary.slices, b.stationary.slices)
    mov_dim, mov_merged = check_adjacent(a.moving.slices, b.moving.slices)
    result: tuple[str, int, tuple[int, int]] = ("stationary", stat_dim, stat_merged)
    if stat_dim < 0:
        result = ("moving", mov_dim, mov_merged)
    return result


def _activation_merge_info(a: NKIActivation, b: NKIActivation) -> tuple[str, int, tuple[int, int]]:
    """Find the adjacent dimension for two activations."""
    dim, merged = check_adjacent(a.src.slices, b.src.slices)
    return ("src", dim, merged)


def _operand_start(stmt: NKIOp, operand: str, dim: int) -> int:
    """Get the start position of an operand on a given dimension."""
    ref = getattr(stmt, operand)
    return ref.slices[dim][0]


def _output_dim_for_operand(stmt_cls: type, operand: str, op_dim: int) -> int:
    """Map operand field + dimension to the corresponding output dimension."""
    axes_key = _FIELD_TO_AXES.get(operand, operand)
    axis_name = stmt_cls.OPERAND_AXES[axes_key][op_dim]
    return stmt_cls.OUTPUT_AXES.index(axis_name)


def _find_alloc_idx(block: NKIBlock, name: str) -> int:
    """Find the NKIAlloc stmt index for a given tensor name."""
    result = -1
    for i, s in enumerate(block.body):
        if isinstance(s, NKIAlloc) and s.dst == name:
            result = i
            break
    return result
