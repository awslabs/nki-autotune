"""Commute a logical transpose through its PSUM-to-SBUF tensor copy."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.tree import ISANode
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.transpose import NKITranspose
from nkigym.search.state_facts import operation_facts
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite
from nkigym.transforms.helper.canonical_rewrite import finalize_rewrite, replace_buffer
from nkigym.transforms.helper.transpose_pattern import TransposeChain, match_transpose_chain


@dataclass(frozen=True)
class TransposeThroughTensorCopyOption(TransformOption):
    """Commute the transpose block at ``transpose_nid`` through its drain."""

    transpose_nid: int


class TransposeThroughTensorCopy(Transform[TransposeThroughTensorCopyOption]):
    """Move one drained logical transpose from Tensor Engine to DMA."""

    def analyze(self, ir: KernelIR) -> list[TransposeThroughTensorCopyOption]:
        """Return every logical transpose eligible for DMA execution."""
        facts = operation_facts(ir)
        if not facts.has_copy or NKITranspose not in facts.op_classes:
            return []
        options: list[TransposeThroughTensorCopyOption] = []
        root_children = ir.tree.children(ir.tree.root)
        for index, block_nid in enumerate(root_children[:-1]):
            if match_transpose_chain(ir, block_nid, root_children[index + 1], adjacent=True) is not None:
                options.append(TransposeThroughTensorCopyOption(transpose_nid=block_nid))
        return options

    def apply(self, ir: KernelIR, option: TransposeThroughTensorCopyOption) -> KernelIR:
        """Recheck ``option`` and change only the transpose execution engine."""
        match = _match(ir, option)
        if match is None:
            raise TransformLegalityError(
                f"TransposeThroughTensorCopy target {option.transpose_nid} is not an eligible logical transpose"
            )
        new_ir = copy_for_rewrite(ir)
        _apply_match(new_ir, match)
        finalize_rewrite(new_ir)
        return new_ir


def _match(ir: KernelIR, option: TransposeThroughTensorCopyOption) -> TransposeChain | None:
    """Return the logical transpose named by ``option``."""
    result: TransposeChain | None = None
    root_children = ir.tree.children(ir.tree.root)
    if option.transpose_nid in root_children:
        index = root_children.index(option.transpose_nid)
        if index + 1 < len(root_children):
            result = match_transpose_chain(ir, option.transpose_nid, root_children[index + 1], adjacent=True)
    return result


def _apply_match(ir: KernelIR, match: TransposeChain) -> None:
    """Execute one concrete transpose in SBUF while retaining its drain."""
    transpose = ir.tree.isa(match.transpose_leaf)
    source_region = transpose.operand_bindings["data"]
    output_region = transpose.operand_bindings["dst"]
    replace_buffer(ir, replace(ir.buffer(match.psum), location="sbuf"))
    ir.tree.graph.nodes[match.transpose_leaf]["data"] = ISANode(
        op_cls=NKIDMATranspose, operand_bindings={"src": source_region, "dst": output_region}, kwargs={}
    )


__all__ = ["TransposeThroughTensorCopy", "TransposeThroughTensorCopyOption"]
