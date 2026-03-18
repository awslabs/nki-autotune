"""Data reuse transform for NKIKernel: deduplicate identical DMA loads.

Scans all blocks for NKIDmaCopy statements loading from HBM (src.name in
kernel.params). Groups by (src.name, src.slices). Emits TransformOption
for each pair with identical source. Cross-block pairs trigger block
concatenation via resolve_option.

Block-level analysis is cached by ``id(block)`` so unchanged blocks
(same Python object across transforms) skip full body scans.

Pair enumeration is lazy: ``_LazyPairSeq`` stores ref groups and computes
C(n,2) pairs on demand via combinatorial unranking, avoiding creation of
millions of TransformOption objects that are never accessed.
"""

import bisect
import dataclasses
from collections.abc import Sequence
from math import isqrt
from typing import overload

from nkigym.codegen.types import NKIBlock, NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import NKIOp
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.transforms.base import NKITransform, StmtRef, TransformOption
from nkigym.transforms.block_merge import replace_block, resolve_option


class _LazyPairSeq(Sequence[TransformOption]):
    """Virtual sequence of TransformOption pairs from ref groups.

    Computes pairs on demand via combinatorial unranking, preserving
    the same ordering as ``itertools.combinations(refs, 2)``.
    Avoids materializing millions of TransformOption objects.
    """

    def __init__(self, groups: list[list[StmtRef]]) -> None:
        """Build offset table from ref groups.

        Args:
            groups: List of ref lists, each with 2+ refs.
        """
        offsets: list[int] = []
        total = 0
        for refs in groups:
            offsets.append(total)
            n = len(refs)
            total += n * (n - 1) // 2
        self._groups = groups
        self._offsets = tuple(offsets)
        self._total = total

    def __len__(self) -> int:
        """Total number of pairs across all groups."""
        return self._total

    @overload
    def __getitem__(self, idx: int) -> TransformOption:
        """Get a single pair by flat index."""
        ...

    @overload
    def __getitem__(self, idx: slice) -> list[TransformOption]:
        """Get a range of pairs by slice."""
        ...

    def __getitem__(self, idx: int | slice) -> TransformOption | list[TransformOption]:
        """Compute the idx-th pair via combinatorial unranking."""
        result: TransformOption | list[TransformOption]
        if isinstance(idx, slice):
            result = [self[i] for i in range(*idx.indices(len(self)))]
        else:
            seg = bisect.bisect_right(self._offsets, idx) - 1
            result = _unrank_pair(self._groups[seg], idx - self._offsets[seg])
        return result

    def __eq__(self, other: object) -> bool:
        """Element-wise comparison with lists (for tests)."""
        result = False
        if isinstance(other, (list, _LazyPairSeq)):
            result = len(self) == len(other) and all(self[i] == other[i] for i in range(len(self)))
        return result


def _unrank_pair(refs: list[StmtRef], local: int) -> TransformOption:
    """Unrank a combination index to a TransformOption pair.

    Given refs of length n, maps flat index ``local`` (in [0, C(n,2)))
    to the corresponding pair in ``combinations(refs, 2)`` order.

    Args:
        refs: List of statement references.
        local: Flat index within this group's pairs.

    Returns:
        TransformOption for the pair at position ``local``.
    """
    n = len(refs)
    discriminant = (2 * n - 1) ** 2 - 8 * local
    i = (2 * n - 1 - isqrt(discriminant)) // 2
    row_start = i * (2 * n - i - 1) // 2
    if row_start > local:
        i -= 1
        row_start = i * (2 * n - i - 1) // 2
    j = i + 1 + (local - row_start)
    return TransformOption(refs[i], refs[j])


class DataReuseTransform(NKITransform):
    """Deduplicate identical DMA loads across blocks.

    ``analyze()`` finds pairs of DMA loads with identical HBM source.
    ``apply()`` removes the duplicate and renames downstream refs.
    Per-block analysis is cached by ``id(block)`` to skip unchanged blocks.
    """

    name = "data_reuse"

    def __init__(self) -> None:
        """Initialize with empty block analysis cache."""
        self._block_cache: dict[int, list] = {}

    def analyze(self, kernel: NKIKernel) -> _LazyPairSeq:
        """Find pairs of DMA loads with identical HBM source slices.

        Args:
            kernel: The NKI kernel to analyze.

        Returns:
            Lazy sequence of TransformOption pairs.
        """
        return _find_reuse_pairs(kernel, self._block_cache)

    def apply(self, kernel: NKIKernel, option: TransformOption) -> NKIKernel:
        """Remove a duplicate DMA load and rename downstream references.

        Args:
            kernel: The NKI kernel to transform.
            option: A TransformOption from ``analyze()``.

        Returns:
            New NKIKernel with the duplicate removed.
        """
        return _apply_reuse(kernel, option)


def _scan_block_dma(block: NKIBlock, hbm_names: set[str]) -> list:
    """Scan a block for HBM DMA loads, return (key, StmtRef) entries.

    Args:
        block: The block to scan.
        hbm_names: Set of HBM parameter names.

    Returns:
        List of (grouping_key, stmt_ref) entries for HBM DMA loads.
    """
    entries: list = []
    for si, stmt in enumerate(block.body):
        if isinstance(stmt, NKIDmaCopy) and stmt.src.name in hbm_names:
            entries.append(((stmt.src.name, stmt.src.slices), StmtRef(block.name, si)))
    return entries


def _find_reuse_pairs(kernel: NKIKernel, cache: dict[int, list]) -> _LazyPairSeq:
    """Scan blocks for duplicate DMA loads, using per-block cache.

    Returns a lazy sequence that computes C(n,2) pairs on demand via
    combinatorial unranking — same ordering as combinations(refs, 2).

    Args:
        kernel: The NKI kernel.
        cache: Per-block cache mapping id(block) to DMA entries.

    Returns:
        Lazy sequence of TransformOption pairs for identical loads.
    """
    hbm_names = set(kernel.params)
    ref_groups: dict[tuple, list[StmtRef]] = {}
    for block in kernel.blocks:
        bid = id(block)
        entries = cache.get(bid)
        if entries is None:
            entries = _scan_block_dma(block, hbm_names)
            cache[bid] = entries
        for key, ref in entries:
            ref_groups.setdefault(key, []).append(ref)
    groups = [refs for refs in ref_groups.values() if len(refs) >= 2]
    return _LazyPairSeq(groups)


def _apply_reuse(kernel: NKIKernel, option: TransformOption) -> NKIKernel:
    """Remove a duplicate DMA load and rename all downstream consumers.

    Args:
        kernel: The NKI kernel.
        option: The reuse pair to apply.

    Returns:
        New kernel with duplicate removed.
    """
    result_kernel, block, idx_a, idx_b = resolve_option(kernel, option)
    stmt_a = block.body[idx_a]
    stmt_b = block.body[idx_b]
    assert isinstance(stmt_a, NKIDmaCopy)
    assert isinstance(stmt_b, NKIDmaCopy)
    keep_name = stmt_a.dst.name
    drop_name = stmt_b.dst.name
    new_body = _remove_and_rename(block.body, idx_b, drop_name, keep_name)
    new_block = block._replace(body=new_body)
    return replace_block(result_kernel, block.name, new_block)


def _remove_and_rename(body: tuple, drop_idx: int, old_name: str, new_name: str) -> tuple:
    """Remove statement at drop_idx and rename consumer references.

    Only renames TensorRef fields (consumers), not NKIAlloc dst strings
    (producers). Orphaned allocs are handled by DCE.

    Args:
        body: Block body tuple.
        drop_idx: Statement index to remove.
        old_name: Tensor name to replace in consumers.
        new_name: Replacement tensor name.

    Returns:
        New body tuple with statement removed and consumer refs renamed.
    """
    filtered = [s for i, s in enumerate(body) if i != drop_idx]
    rename_map = {old_name: new_name}
    return tuple(_rename_refs(s, rename_map) for s in filtered)


def _rename_refs(stmt: NKIOp, rename_map: dict[str, str]) -> NKIOp:
    """Rename TensorRef fields only, leaving NKIAlloc dst strings untouched.

    Args:
        stmt: An NKI statement.
        rename_map: Mapping from old names to new names.

    Returns:
        New statement with renamed TensorRef fields.
    """
    kwargs: dict[str, object] = {}
    changed = False
    for fld in dataclasses.fields(stmt):
        val = getattr(stmt, fld.name)
        if isinstance(val, TensorRef) and rename_map.get(val.name, val.name) != val.name:
            kwargs[fld.name] = TensorRef(rename_map[val.name], val.shape, val.slices)
            changed = True
        else:
            kwargs[fld.name] = val
    result = type(stmt)(**kwargs) if changed else stmt
    return result
