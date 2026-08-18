"""Commute a logical transpose through its PSUM-to-SBUF tensor copy."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.tree import ISANode
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
from nkigym.search.buffer_placement import layout_satisfies_alignment
from nkigym.search.state_facts import operation_facts
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite
from nkigym.transforms.helper.canonical_rewrite import finalize_rewrite, is_canonical_block, replace_buffer, single_leaf
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
        if not facts.has_copy or not ({NKITranspose, NKIDMATranspose} & facts.op_classes):
            return []
        options: list[TransposeThroughTensorCopyOption] = []
        root_children = ir.tree.children(ir.tree.root)
        for index, block_nid in enumerate(root_children[:-1]):
            option = TransposeThroughTensorCopyOption(transpose_nid=block_nid)
            if _match(ir, option)[0] is not None:
                options.append(option)
        return options

    def apply(self, ir: KernelIR, option: TransposeThroughTensorCopyOption) -> KernelIR:
        """Recheck ``option`` and change only the transpose execution engine."""
        match, reverse = _match(ir, option)
        if match is None:
            raise TransformLegalityError(
                f"TransposeThroughTensorCopy target {option.transpose_nid} is not an eligible logical transpose"
            )
        new_ir = copy_for_rewrite(ir)
        _apply_match(new_ir, match, reverse)
        finalize_rewrite(new_ir)
        return new_ir


def _match(ir: KernelIR, option: TransposeThroughTensorCopyOption) -> tuple[TransposeChain | None, bool]:
    """Return the logical transpose named by ``option``."""
    result: TransposeChain | None = None
    reverse = False
    root_children = ir.tree.children(ir.tree.root)
    if option.transpose_nid in root_children:
        index = root_children.index(option.transpose_nid)
        if index + 1 < len(root_children):
            result = match_transpose_chain(ir, option.transpose_nid, root_children[index + 1], adjacent=True)
            if result is not None:
                candidate = replace(ir.buffer(result.psum), location="sbuf")
                alignment = NKIDMATranspose.OUTPUT_TILE_ALIGNMENT_BYTES["dst"]
                if not layout_satisfies_alignment(candidate, alignment):
                    result = None
            else:
                result = _match_dma_transpose_chain(ir, option.transpose_nid, root_children[index + 1])
                reverse = result is not None
    return result, reverse


def _match_dma_transpose_chain(ir: KernelIR, transpose_block: int, drain_block: int) -> TransposeChain | None:
    """Return one isolated DMA-transpose and tensor-copy chain."""
    result: TransposeChain | None = None
    if is_canonical_block(ir, transpose_block) and is_canonical_block(ir, drain_block):
        transpose_leaf = single_leaf(ir.tree, transpose_block)
        drain_leaf = single_leaf(ir.tree, drain_block)
        if transpose_leaf is not None and drain_leaf is not None:
            transpose = ir.tree.isa(transpose_leaf)
            drain = ir.tree.isa(drain_leaf)
            if transpose.op_cls is NKIDMATranspose and drain.op_cls is NKITensorCopy:
                source = transpose.operand_bindings["src"].tensor
                intermediate = transpose.operand_bindings["dst"].tensor
                output = drain.operand_bindings["dst"].tensor
                source_buffer = ir.buffer(source)
                intermediate_buffer = ir.buffer(intermediate)
                output_buffer = ir.buffer(output)
                axes = ir.tree.block(transpose_block).axis_map
                drain_axes = ir.tree.block(drain_block).axis_map
                valid = (
                    drain.operand_bindings["src"].tensor == intermediate
                    and source_buffer.shape[::-1] == intermediate_buffer.shape == output_buffer.shape
                    and source_buffer.location == intermediate_buffer.location == output_buffer.location == "sbuf"
                    and source_buffer.dtype == intermediate_buffer.dtype == output_buffer.dtype
                    and NKITranspose.accepts_input_storage_dtypes({"data": source_buffer.physical_dtype()})
                    and source_buffer.physical_dtype() == intermediate_buffer.physical_dtype()
                    and drain_axes.get("P") == axes.get("F")
                    and drain_axes.get("F") == axes.get("P")
                    and set(ir.dependency.touches_by_tensor.get(intermediate, ())) == {transpose_leaf, drain_leaf}
                )
                if valid and isinstance(axes.get("P"), str) and isinstance(axes.get("F"), str):
                    result = TransposeChain(
                        transpose_block=transpose_block,
                        drain_block=drain_block,
                        transpose_leaf=transpose_leaf,
                        drain_leaf=drain_leaf,
                        source=source,
                        psum=intermediate,
                        output=output,
                        source_axes=(axes["P"], axes["F"]),
                    )
    return result


def _apply_match(ir: KernelIR, match: TransposeChain, reverse: bool) -> None:
    """Execute one concrete transpose in SBUF while retaining its drain."""
    transpose = ir.tree.isa(match.transpose_leaf)
    source_slot = "src" if reverse else "data"
    source_region = transpose.operand_bindings[source_slot]
    output_region = transpose.operand_bindings["dst"]
    replace_buffer(ir, replace(ir.buffer(match.psum), location="psum" if reverse else "sbuf"))
    ir.tree.graph.nodes[match.transpose_leaf]["data"] = ISANode(
        op_cls=NKITranspose if reverse else NKIDMATranspose,
        operand_bindings={("data" if reverse else "src"): source_region, "dst": output_region},
        kwargs={},
    )


__all__ = ["TransposeThroughTensorCopy", "TransposeThroughTensorCopyOption"]
