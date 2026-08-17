"""Shared tree-mutation helpers for transforms."""

from __future__ import annotations

from dataclasses import replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const
from nkigym.ir.tree import BlockNode, Buffer, ISANode, KernelTree
from nkigym.transforms.base import invalidate_software_pipeline_overlap


def _replace_in_parent_children(
    tree: KernelTree, parent_nid: int, old_children: list[int], new_children: list[int]
) -> None:
    """Replace a contiguous child slice while preserving sibling order."""
    siblings_before = tree.children(parent_nid)
    start = siblings_before.index(old_children[0])
    assert siblings_before[start : start + len(old_children)] == list(old_children), (
        f"_replace_in_parent_children: old_children {old_children} is not a contiguous "
        f"slice of parent_nid={parent_nid} children {siblings_before}"
    )
    new_order = siblings_before[:start] + list(new_children) + siblings_before[start + len(old_children) :]
    tree.graph.remove_edges_from((parent_nid, child) for child in siblings_before)
    tree.graph.add_edges_from((parent_nid, child) for child in new_order)


def _block_local_descendants(tree: KernelTree, block_nid: int) -> list[int]:
    """Return descendants in the block's iteration scope without sub-blocks."""
    result: list[int] = []
    stack = [block_nid]
    while stack:
        for child in tree.children(stack.pop()):
            if not isinstance(tree.data(child), BlockNode):
                result.append(child)
                stack.append(child)
    return result


def invalidate_stale_software_pipelines(ir: KernelIR, invalidated_loop_nids: frozenset[int] = frozenset()) -> None:
    """Drop pipeline metadata whose staged structure changed."""
    invalidate_software_pipeline_overlap(ir.tree)
    active_versioned: set[str] = set()
    for block_nid in list(ir.tree.blocks()):
        block = ir.tree.block(block_nid)
        annotation = block.annotations.get("software_pipeline")
        if annotation is None:
            continue
        loop_nid = annotation["loop_nid"]
        expected_children = annotation.get("children")
        stale = loop_nid in invalidated_loop_nids or loop_nid not in ir.tree.graph
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

    version_changes: dict[str, tuple[Buffer, Buffer]] = {}
    for block_nid in ir.tree.blocks():
        block = ir.tree.block(block_nid)
        allocations: list[Buffer] = []
        for buffer in block.alloc_buffers:
            updated = buffer
            if buffer.versions > 1 and buffer.name not in active_versioned:
                updated = replace(buffer, versions=1)
                version_changes[buffer.name] = (buffer, updated)
            allocations.append(updated)
        if tuple(allocations) != block.alloc_buffers:
            ir.tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=tuple(allocations))
    _rebase_access_pattern_strides(ir, version_changes)


def _rebase_access_pattern_strides(ir: KernelIR, version_changes: dict[str, tuple[Buffer, Buffer]]) -> None:
    """Rebase physical partition strides after pipeline versions are removed."""
    for nid in ir.tree.preorder():
        node = ir.tree.data(nid)
        if not isinstance(node, ISANode) or not node.access_patterns:
            continue
        patterns = dict(node.access_patterns)
        for slot, pattern in node.access_patterns.items():
            tensor = node.operand_bindings[slot].tensor
            buffers = version_changes.get(tensor)
            if buffers is None:
                continue
            old_buffer, new_buffer = buffers
            old_shape = old_buffer.per_tile_physical_shape()
            new_shape = new_buffer.per_tile_physical_shape()
            old_stride = Const(value=old_shape[1] * old_shape[2])
            new_stride = Const(value=new_shape[1] * new_shape[2])
            partition_extent = Const(value=old_shape[0])
            dimensions = tuple(
                (new_stride, extent) if stride == old_stride and extent == partition_extent else (stride, extent)
                for stride, extent in pattern.pattern
            )
            if dimensions == pattern.pattern:
                raise AssertionError(f"{tensor}: access pattern has no physical partition stride {old_stride.value}")
            patterns[slot] = replace(pattern, pattern=dimensions)
        if patterns != node.access_patterns:
            ir.tree.graph.nodes[nid]["data"] = replace(node, access_patterns=patterns)


__all__ = ["_block_local_descendants", "_replace_in_parent_children", "invalidate_stale_software_pipelines"]
