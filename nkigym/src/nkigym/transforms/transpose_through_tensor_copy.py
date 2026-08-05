"""Commute a logical transpose through its PSUM-to-SBUF tensor copy."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.tree import ISANode
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.transforms._canonical_rewrite import finalize_rewrite, remove_buffers
from nkigym.transforms._transpose_pattern import TransposeChain, match_transpose_chain
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class TransposeThroughTensorCopyOption(TransformOption):
    """Commute the transpose block at ``transpose_nid`` through its drain."""

    transpose_nid: int


class TransposeThroughTensorCopy(Transform[TransposeThroughTensorCopyOption]):
    """Apply ``Copy(T(x)) = DMATranspose(x)`` to one logical transpose."""

    def analyze(self, ir: KernelIR) -> list[TransposeThroughTensorCopyOption]:
        """Return every logical transpose eligible for DMA execution."""
        options: list[TransposeThroughTensorCopyOption] = []
        root_children = ir.tree.children(ir.tree.root)
        for index, block_nid in enumerate(root_children[:-1]):
            if match_transpose_chain(ir, block_nid, root_children[index + 1]) is not None:
                options.append(TransposeThroughTensorCopyOption(transpose_nid=block_nid))
        return options

    def apply(self, ir: KernelIR, option: TransposeThroughTensorCopyOption) -> KernelIR:
        """Recheck ``option`` and replace the transpose/drain with DMA transpose."""
        match = _match(ir, option)
        if match is None:
            raise TransformLegalityError(
                f"TransposeThroughTensorCopy target {option.transpose_nid} is not an eligible logical transpose"
            )
        new_ir = copy.deepcopy(ir)
        copied_match = _match(new_ir, option)
        if copied_match is None:
            raise AssertionError("TransposeThroughTensorCopy match disappeared after deepcopy")
        _apply_match(new_ir, copied_match)
        finalize_rewrite(new_ir)
        return new_ir


def _match(ir: KernelIR, option: TransposeThroughTensorCopyOption) -> TransposeChain | None:
    """Return the logical transpose named by ``option``."""
    result: TransposeChain | None = None
    root_children = ir.tree.children(ir.tree.root)
    if option.transpose_nid in root_children:
        index = root_children.index(option.transpose_nid)
        if index + 1 < len(root_children):
            result = match_transpose_chain(ir, option.transpose_nid, root_children[index + 1])
    return result


def _apply_match(ir: KernelIR, match: TransposeChain) -> None:
    """Replace one concrete transpose/drain chain in place."""
    transpose = ir.tree.isa(match.transpose_leaf)
    source_region = transpose.operand_bindings["data"]
    output_region = replace(transpose.operand_bindings["dst"], tensor=match.output)
    block = ir.tree.block(match.transpose_block)
    ir.tree.graph.nodes[match.transpose_block]["data"] = replace(block, reads=(source_region,), writes=(output_region,))
    ir.tree.graph.nodes[match.transpose_leaf]["data"] = ISANode(
        op_cls=NKIDMATranspose, operand_bindings={"src": source_region, "dst": output_region}, kwargs={}
    )
    _remove_drain_block(ir, match.transpose_block, match.drain_block)
    remove_buffers(ir, {match.psum})


def _remove_drain_block(ir: KernelIR, retained_block: int, drain_block: int) -> None:
    """Transfer declarations, then remove one top-level drain block."""
    retained = ir.tree.block(retained_block)
    drain = ir.tree.block(drain_block)
    ir.tree.graph.nodes[retained_block]["data"] = replace(
        retained, alloc_buffers=(*retained.alloc_buffers, *drain.alloc_buffers)
    )
    _replace_in_parent_children(ir.tree, ir.tree.root, [drain_block], [])
    ir.tree.graph.remove_nodes_from({drain_block, *ir.tree.descendants(drain_block)})


__all__ = ["TransposeThroughTensorCopy", "TransposeThroughTensorCopyOption"]
