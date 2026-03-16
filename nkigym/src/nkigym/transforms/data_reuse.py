"""Data reuse transform for NKIKernel: deduplicate identical DMA loads.

Scans all blocks for NKIDmaCopy statements loading from HBM (src.name in
kernel.params). Groups by (src.name, src.slices). Emits TransformOption
for each pair with identical source. Cross-block pairs trigger block
concatenation via resolve_option.
"""

import dataclasses
from itertools import combinations

from nkigym.codegen.types import NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import NKIOp
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.transforms.base import NKITransform, StmtRef, TransformOption
from nkigym.transforms.block_merge import replace_block, resolve_option


class DataReuseTransform(NKITransform):
    """Deduplicate identical DMA loads across blocks.

    ``analyze()`` finds pairs of DMA loads with identical HBM source.
    ``apply()`` removes the duplicate and renames downstream refs.
    """

    name = "data_reuse"

    def analyze(self, kernel: NKIKernel) -> list[TransformOption]:
        """Find pairs of DMA loads with identical HBM source slices.

        Args:
            kernel: The NKI kernel to analyze.

        Returns:
            List of TransformOption pairs.
        """
        return _find_reuse_pairs(kernel)

    def apply(self, kernel: NKIKernel, option: TransformOption) -> NKIKernel:
        """Remove a duplicate DMA load and rename downstream references.

        Args:
            kernel: The NKI kernel to transform.
            option: A TransformOption from ``analyze()``.

        Returns:
            New NKIKernel with the duplicate removed.
        """
        return _apply_reuse(kernel, option)


def _find_reuse_pairs(kernel: NKIKernel) -> list[TransformOption]:
    """Scan all blocks for duplicate DMA loads from HBM parameters.

    Args:
        kernel: The NKI kernel.

    Returns:
        List of TransformOption pairs for identical loads.
    """
    hbm_names = set(kernel.params)
    groups: dict[tuple[str, tuple[tuple[int, int], ...]], list[StmtRef]] = {}
    for block in kernel.blocks:
        for si, stmt in enumerate(block.body):
            if not isinstance(stmt, NKIDmaCopy):
                continue
            if stmt.src.name not in hbm_names:
                continue
            key = (stmt.src.name, stmt.src.slices)
            groups.setdefault(key, []).append(StmtRef(block.name, si))
    pairs: list[TransformOption] = []
    for refs in groups.values():
        if len(refs) >= 2:
            pairs.extend(TransformOption(a, b) for a, b in combinations(refs, 2))
    return pairs


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
