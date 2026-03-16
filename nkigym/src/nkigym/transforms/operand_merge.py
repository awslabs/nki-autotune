"""Operand merge transform for NKIKernel: DMA merge + compute merge.

Finds pairs of adjacent DMA loads or compute stmts that can be merged
into one wider operation. DMA merge combines two DMA loads with adjacent
HBM sources, widening their destination allocs. Compute merge combines
two matmul or activation stmts with adjacent operand slices on one axis.
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
    """Merge adjacent DMA loads or compute stmts into wider operations."""

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
    """Collect DMA merge and compute merge pairs from all blocks."""
    pairs: list[TransformOption] = []
    for block in kernel.blocks:
        pairs.extend(_find_dma_pairs(block))
        pairs.extend(_find_compute_pairs(block))
    return pairs


def _find_dma_pairs(block: NKIBlock) -> list[TransformOption]:
    """Find DMA load pairs with adjacent HBM sources in one block.

    Groups DMA loads by (alloc.dtype, alloc.buffer, dma.src.name).
    Checks pairs for adjacent HBM source slices. Emits options with
    DMA stmt indices (not alloc indices).
    """
    dma_info = _build_dma_alloc_map(block)
    groups: dict[tuple, list[tuple[int, NKIDmaCopy]]] = {}
    for dma_idx, (dma, alloc) in dma_info.items():
        key = (alloc.dtype, alloc.buffer, dma.src.name)
        groups.setdefault(key, []).append((dma_idx, dma))
    pairs: list[TransformOption] = []
    for members in groups.values():
        for (i, dma_a), (j, dma_b) in combinations(members, 2):
            dim, _ = check_adjacent(dma_a.src.slices, dma_b.src.slices)
            if dim >= 0:
                pairs.append(TransformOption(StmtRef(block.name, i), StmtRef(block.name, j)))
    return pairs


def _build_dma_alloc_map(block: NKIBlock) -> dict[int, tuple[NKIDmaCopy, NKIAlloc]]:
    """Map DMA stmt index to (dma_stmt, alloc_stmt).

    Only includes DMAs that are the sole load into an allocated buffer.
    """
    alloc_by_name: dict[str, NKIAlloc] = {}
    for s in block.body:
        if isinstance(s, NKIAlloc):
            alloc_by_name[s.dst] = s
    seen_allocs: set[str] = set()
    result: dict[int, tuple[NKIDmaCopy, NKIAlloc]] = {}
    for i, s in enumerate(block.body):
        if isinstance(s, NKIDmaCopy) and s.dst.name in alloc_by_name and s.dst.name not in seen_allocs:
            seen_allocs.add(s.dst.name)
            result[i] = (s, alloc_by_name[s.dst.name])
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
    """Dispatch merge by statement type: DMA merge or compute merge."""
    result_kernel, block, idx_a, idx_b = resolve_option(kernel, option)
    stmt_a = block.body[idx_a]
    new_block = block
    if isinstance(stmt_a, NKIDmaCopy):
        new_block = _apply_dma_merge(block, idx_a, idx_b)
    elif isinstance(stmt_a, (NKIMatmul, NKIActivation)):
        new_block = _apply_compute_merge(block, idx_a, idx_b)
    return replace_block(result_kernel, block.name, new_block)


def _apply_dma_merge(block: NKIBlock, idx_a: int, idx_b: int) -> NKIBlock:
    """Merge two DMA loads and widen their allocs, remap consumers.

    Always merges the later body-position DMA into the earlier one.
    The late alloc is kept as dead code for DCE. Computes per-name
    offsets from source slice ordering to handle cases where early
    body position has a higher source range.
    """
    if idx_a > idx_b:
        idx_a, idx_b = idx_b, idx_a
    dma_a = block.body[idx_a]
    dma_b = block.body[idx_b]
    assert isinstance(dma_a, NKIDmaCopy)
    assert isinstance(dma_b, NKIDmaCopy)
    alloc_a_idx = _find_alloc_idx(block, dma_a.dst.name)
    alloc_a = block.body[alloc_a_idx]
    alloc_b = block.body[_find_alloc_idx(block, dma_b.dst.name)]
    assert isinstance(alloc_a, NKIAlloc)
    assert isinstance(alloc_b, NKIAlloc)
    dim, merged = check_adjacent(dma_a.src.slices, dma_b.src.slices)
    a_first = dma_a.src.slices[dim][0] < dma_b.src.slices[dim][0]
    early_offset = 0 if a_first else alloc_b.shape[dim]
    late_offset = alloc_a.shape[dim] if a_first else 0
    new_size = alloc_a.shape[dim] + alloc_b.shape[dim]
    new_shape = (*alloc_a.shape[:dim], new_size, *alloc_a.shape[dim + 1 :])
    new_alloc = NKIAlloc(dst=alloc_a.dst, shape=new_shape, dtype=alloc_a.dtype, buffer=alloc_a.buffer)
    new_dma = _widen_dma(dma_a, new_shape, dim, merged)
    drop = {idx_b}
    replace: dict[int, NKIOp] = {alloc_a_idx: new_alloc, idx_a: new_dma}
    return _rebuild_body(block, drop, replace, alloc_a.dst, alloc_b.dst, dim, early_offset, late_offset)


def _apply_compute_merge(block: NKIBlock, idx_a: int, idx_b: int) -> NKIBlock:
    """Merge two compute stmts with adjacent operand slices.

    Always merges the later body-position stmt into the earlier one.
    When stmts write to different allocs: widen early, drop late, shift
    consumers. When stmts share an alloc: preserve it, no shifting.
    """
    if idx_a > idx_b:
        idx_a, idx_b = idx_b, idx_a
    stmt_a = block.body[idx_a]
    stmt_b = block.body[idx_b]
    assert isinstance(stmt_a, (NKIMatmul, NKIActivation))
    assert isinstance(stmt_b, (NKIMatmul, NKIActivation))
    operand, op_dim, merged_range = _compute_merge_info(stmt_a, stmt_b)
    out_dim = _output_dim_for_operand(type(stmt_a), operand, op_dim)
    a_size = stmt_a.dst.slices[out_dim][1] - stmt_a.dst.slices[out_dim][0]
    b_size = stmt_b.dst.slices[out_dim][1] - stmt_b.dst.slices[out_dim][0]
    same_alloc = stmt_a.dst.name == stmt_b.dst.name
    a_first = _operand_start(stmt_a, operand, op_dim) < _operand_start(stmt_b, operand, op_dim)
    early_offset = 0 if same_alloc or a_first else b_size
    late_offset = 0 if same_alloc else (a_size if a_first else 0)
    min_start = min(stmt_a.dst.slices[out_dim][0], stmt_b.dst.slices[out_dim][0]) if same_alloc else 0
    out_merged = (min_start, min_start + a_size + b_size)
    new_stmt = _build_merged_compute(stmt_a, stmt_b, a_first, operand, op_dim, merged_range, out_dim, out_merged)
    replace, drop = _compute_merge_replacements(block, idx_a, idx_b, stmt_a, new_stmt, out_dim, out_merged, same_alloc)
    return _rebuild_body(block, drop, replace, stmt_a.dst.name, stmt_b.dst.name, out_dim, early_offset, late_offset)


_ComputeStmt = NKIMatmul | NKIActivation


def _build_merged_compute(
    stmt_a: _ComputeStmt,
    stmt_b: _ComputeStmt,
    a_first: bool,
    operand: str,
    op_dim: int,
    merged_range: tuple[int, int],
    out_dim: int,
    out_merged: tuple[int, int],
) -> _ComputeStmt:
    """Build the widened compute stmt, ensuring dst uses early's name."""
    base_stmt = stmt_a if a_first else stmt_b
    new_stmt = _widen_compute(base_stmt, operand, op_dim, merged_range, out_dim, out_merged)
    assert isinstance(new_stmt, (NKIMatmul, NKIActivation))
    if new_stmt.dst.name != stmt_a.dst.name:
        new_dst = TensorRef(stmt_a.dst.name, new_stmt.dst.shape, new_stmt.dst.slices)
        new_stmt = dataclasses.replace(new_stmt, dst=new_dst)
    return new_stmt


def _compute_merge_replacements(
    block: NKIBlock,
    idx_a: int,
    idx_b: int,
    stmt_a: _ComputeStmt,
    new_stmt: _ComputeStmt,
    out_dim: int,
    out_merged: tuple[int, int],
    same_alloc: bool,
) -> tuple[dict[int, NKIOp], set[int]]:
    """Build replace and drop dicts for compute merge.

    When stmts use different allocs: widen early alloc, keep late alloc
    as dead code for DCE. When stmts share an alloc: leave unchanged.
    """
    replace: dict[int, NKIOp] = {idx_a: new_stmt}
    drop: set[int] = {idx_b}
    if not same_alloc:
        early_alloc_idx = _find_alloc_idx(block, stmt_a.dst.name)
        if early_alloc_idx >= 0:
            old_alloc = block.body[early_alloc_idx]
            assert isinstance(old_alloc, NKIAlloc)
            new_shape = (*old_alloc.shape[:out_dim], out_merged[1], *old_alloc.shape[out_dim + 1 :])
            replace[early_alloc_idx] = NKIAlloc(
                dst=stmt_a.dst.name, shape=new_shape, dtype=old_alloc.dtype, buffer=old_alloc.buffer
            )
    return replace, drop


def _rebuild_body(
    block: NKIBlock,
    drop: set[int],
    replace: dict[int, NKIOp],
    early_name: str,
    late_name: str,
    dim: int,
    early_offset: int,
    late_offset: int,
) -> NKIBlock:
    """Rebuild block body: drop stmts, insert replacements, shift consumers.

    Two-pass consumer update: shift early's consumers (pass 1),
    then rename + shift late's consumers (pass 2). Replacement stmts
    are inserted directly without shifting.
    """
    new_body: list[NKIOp] = []
    for i, s in enumerate(block.body):
        if i in drop:
            continue
        if i in replace:
            new_body.append(replace[i])
        else:
            stmt = _shift_refs(s, early_name, early_name, dim, early_offset)
            stmt = _shift_refs(stmt, late_name, early_name, dim, late_offset)
            new_body.append(stmt)
    return block._replace(body=tuple(new_body))


def _shift_refs(stmt: NKIOp, old_name: str, new_name: str, dim: int, offset: int) -> NKIOp:
    """Rename old_name to new_name and shift slices on dim by offset.

    Only shifts TensorRef fields whose name matches old_name.
    NKIAlloc dst strings are never renamed — dead allocs are left for DCE.
    """
    kwargs: dict[str, object] = {}
    changed = False
    for fld in dataclasses.fields(stmt):
        val = getattr(stmt, fld.name)
        if isinstance(val, TensorRef) and val.name == old_name:
            shifted = _shift_slice(val.slices, dim, offset)
            kwargs[fld.name] = TensorRef(new_name, val.shape, shifted)
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
    src_shape = tuple(e - s for s, e in new_src_slices)
    return NKIDmaCopy(
        dst=TensorRef(dma.dst.name, new_shape, new_dst_slices), src=TensorRef(dma.src.name, src_shape, new_src_slices)
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
