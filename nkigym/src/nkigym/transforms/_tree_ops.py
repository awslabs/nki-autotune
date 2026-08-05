"""Shared low-level tree-mutation helpers for transforms.

The networkx ``DiGraph`` does not preserve sibling order when nodes are
added or removed in arbitrary order — ``add_edge`` always appends to the
predecessor's successor list. Transforms that replace one or more
contiguous children of a parent must therefore rewrite the parent's full
out-edge list to keep sibling order stable.
"""

from __future__ import annotations

from dataclasses import replace

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, KernelTree


def _replace_in_parent_children(
    tree: KernelTree, parent_nid: int, old_children: list[int], new_children: list[int]
) -> None:
    """Replace ``old_children`` with ``new_children`` at the same position in ``parent_nid``'s child list.

    ``old_children`` must be a contiguous slice of ``tree.children(parent_nid)``
    in order. The function wipes ``parent_nid``'s out-edges and re-adds them
    so the new children occupy the slot the old children occupied; all other
    siblings keep their relative order.

    The nodes themselves are not removed from the graph — caller is
    responsible for any subsequent ``remove_node`` cleanup of orphaned old
    children.
    """
    siblings_before = tree.children(parent_nid)
    start = siblings_before.index(old_children[0])
    assert siblings_before[start : start + len(old_children)] == list(old_children), (
        f"_replace_in_parent_children: old_children {old_children} is not a contiguous "
        f"slice of parent_nid={parent_nid} children {siblings_before}"
    )
    new_order = siblings_before[:start] + list(new_children) + siblings_before[start + len(old_children) :]
    for child in siblings_before:
        tree.graph.remove_edge(parent_nid, child)
    for child in new_order:
        tree.graph.add_edge(parent_nid, child)


def _block_local_descendants(tree: KernelTree, block_nid: int) -> list[int]:
    """Yield nids descended from ``block_nid`` that share its iter_var scope.

    Walks the tree but does NOT enter sub-blocks (BlockNodes other than block_nid
    itself, including the block's init sub-block and any nested compute_at-sunk
    blocks). This is the scope over which iter_value substitutions apply when
    a Split / Fuse rewrites this block's bindings.
    """
    from nkigym.ir.tree import BlockNode

    result: list[int] = []
    stack = [block_nid]
    while stack:
        cur = stack.pop()
        for child in tree.children(cur):
            child_data = tree.data(child)
            if isinstance(child_data, BlockNode):
                """Don't descend into sub-blocks; they have their own iter_var space."""
                continue
            result.append(child)
            stack.append(child)
    return result


def invalidate_stale_software_pipelines(ir: KernelIR) -> None:
    """Drop pipeline metadata whose staged child list changed.

    Software-pipeline stages are positional labels for one loop's exact direct
    children. Structural rewrites may remove, replace, reorder, or reparent
    those children. Such a rewrite invalidates the annotation and every buffer
    version owned only by that annotation.
    """
    active_versioned: set[str] = set()
    for block_nid in list(ir.tree.blocks()):
        block = ir.tree.block(block_nid)
        annotation = block.annotations.get("software_pipeline")
        if annotation is None:
            continue
        loop_nid = annotation["loop_nid"]
        expected_children = annotation.get("children")
        stale = loop_nid not in ir.tree.graph
        if not stale and annotation.get("loop") is not None:
            stale = ir.tree.data(loop_nid) != annotation["loop"]
        if not stale and expected_children is not None:
            stale = tuple(ir.tree.children(loop_nid)) != tuple(expected_children)
        if stale:
            annotations = dict(block.annotations)
            del annotations["software_pipeline"]
            ir.tree.graph.nodes[block_nid]["data"] = replace(block, annotations=annotations)
        else:
            active_versioned.update(annotation["versioned_buffers"])

    for block_nid in ir.tree.blocks():
        block = ir.tree.block(block_nid)
        allocations = tuple(
            replace(buffer, versions=1) if buffer.versions > 1 and buffer.name not in active_versioned else buffer
            for buffer in block.alloc_buffers
        )
        if allocations != block.alloc_buffers:
            ir.tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=allocations)


__all__ = ["_block_local_descendants", "_replace_in_parent_children", "invalidate_stale_software_pipelines"]
