"""Loop-carried state helpers for CodeMotion legality."""

from __future__ import annotations

from nkigym.ir import KernelIR
from nkigym.ir.dependency import _access_invariant_across
from nkigym.ir.interval import regions_disjoint
from nkigym.ir.tree import BufferRegion, ISANode


def regions_overlap(
    ir: KernelIR, first_leaf: int, first_region: BufferRegion, second_leaf: int, second_region: BufferRegion
) -> bool:
    """Return whether two regions of one materialized tensor may overlap."""
    extents = {**ir.dependency.info(first_leaf).extents, **ir.dependency.info(second_leaf).extents}
    buffer = ir.buffer(first_region.tensor)
    return not regions_disjoint(first_region, second_region, buffer, buffer, extents)


def loop_carries_plain_state(ir: KernelIR, loop_nid: int, tensor: str, excluded_leaf: int) -> bool:
    """Return whether a read-before-write carries ``tensor`` between iterations."""
    tree = ir.tree
    loop = tree.loop(loop_nid)
    accesses: list[tuple[int, tuple[BufferRegion, ...], tuple[BufferRegion, ...]]] = []
    for leaf in tree.preorder(loop_nid):
        if leaf == excluded_leaf or not isinstance(tree.data(leaf), ISANode):
            continue
        info = ir.dependency.info(leaf)
        reads = tuple(
            region
            for region in info.read_regions
            if region.tensor == tensor and _access_invariant_across(tree, leaf, loop.loop_var, tensor)
        )
        writes = tuple(
            region
            for region in info.write_regions
            if region.tensor == tensor and _access_invariant_across(tree, leaf, loop.loop_var, tensor)
        )
        if reads or writes:
            accesses.append((leaf, reads, writes))

    return any(
        not any(
            regions_overlap(ir, prior_leaf, prior_write, read_leaf, read_region)
            for prior_leaf, _prior_reads, prior_writes in accesses[:read_index]
            for prior_write in prior_writes
        )
        and any(
            regions_overlap(ir, later_leaf, later_write, read_leaf, read_region)
            for later_leaf, _later_reads, later_writes in accesses[read_index:]
            for later_write in later_writes
        )
        for read_index, (read_leaf, read_regions, _read_writes) in enumerate(accesses)
        for read_region in read_regions
    )


__all__ = ["loop_carries_plain_state", "regions_overlap"]
