"""Data reuse transform for NKIKernel: deduplicate identical DMA loads.

Scans all blocks for NKIDmaCopy statements loading from HBM (src.name in
kernel.params). Groups by (src.name, src.slices). Emits TransformOption
for each pair with identical source. Cross-block pairs trigger block
concatenation via resolve_option.
"""

from itertools import combinations

from nkigym.codegen.types import NKIKernel, _rename_stmt
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
    """Remove a statement and rename old_name to new_name in remaining stmts.

    Args:
        body: Block body tuple.
        drop_idx: Index of statement to remove.
        old_name: Tensor name to replace.
        new_name: Replacement tensor name.

    Returns:
        New body tuple with statement removed and names renamed.
    """
    filtered = [s for i, s in enumerate(body) if i != drop_idx]
    rename_map = {old_name: new_name}
    return tuple(_rename_stmt(s, rename_map) for s in filtered)
