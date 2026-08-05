"""Commute one logical transpose upward through a matrix multiplication."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
from nkigym.transforms._canonical_rewrite import (
    canonical_spec,
    finalize_rewrite,
    is_canonical_block,
    remove_buffers,
    replace_buffer,
    required_spec,
    rewrite_block,
    single_leaf,
)
from nkigym.transforms._transpose_pattern import TransposeChain, match_transpose_chain
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class TransposeThroughMatmulOption(TransformOption):
    """Commute the transpose block at ``transpose_nid`` through its producer."""

    transpose_nid: int


@dataclass(frozen=True)
class _Match:
    """A canonical matmul, drain, and following transpose chain."""

    memset_block: int
    matmul_block: int
    matmul_drain_block: int
    transpose: TransposeChain
    stationary: str
    moving: str
    old_psum: str
    old_output: str


class TransposeThroughMatmul(Transform[TransposeThroughMatmulOption]):
    """Apply ``T(A.T @ B) = B.T @ A`` to one adjacent transpose."""

    def analyze(self, ir: KernelIR) -> list[TransposeThroughMatmulOption]:
        """Return every transpose that can commute through a canonical matmul."""
        options: list[TransposeThroughMatmulOption] = []
        for block_nid in ir.tree.children(ir.tree.root):
            option = TransposeThroughMatmulOption(transpose_nid=block_nid)
            if _match(ir, option) is not None:
                options.append(option)
        return options

    def apply(self, ir: KernelIR, option: TransposeThroughMatmulOption) -> KernelIR:
        """Recheck ``option``, swap the matmul operands, and consume the transpose."""
        match = _match(ir, option)
        if match is None:
            raise TransformLegalityError(
                f"TransposeThroughMatmul target {option.transpose_nid} is not adjacent to an eligible canonical matmul"
            )
        new_ir = copy.deepcopy(ir)
        copied_match = _match(new_ir, option)
        if copied_match is None:
            raise AssertionError("TransposeThroughMatmul match disappeared after deepcopy")
        _apply_match(new_ir, copied_match)
        finalize_rewrite(new_ir)
        return new_ir


def _match(ir: KernelIR, option: TransposeThroughMatmulOption) -> _Match | None:
    """Return one legal matmul commute."""
    result: _Match | None = None
    root_children = ir.tree.children(ir.tree.root)
    if option.transpose_nid in root_children:
        index = root_children.index(option.transpose_nid)
        if 3 <= index and index + 1 < len(root_children):
            memset_block = root_children[index - 3]
            matmul_block = root_children[index - 2]
            matmul_drain_block = root_children[index - 1]
            transpose = match_transpose_chain(ir, option.transpose_nid, root_children[index + 1])
            canonical = all(is_canonical_block(ir, block) for block in (memset_block, matmul_block, matmul_drain_block))
            if transpose is not None and canonical:
                result = _validate_segment(
                    ir,
                    memset_block=memset_block,
                    matmul_block=matmul_block,
                    matmul_drain_block=matmul_drain_block,
                    transpose=transpose,
                )
    return result


def _validate_segment(
    ir: KernelIR, *, memset_block: int, matmul_block: int, matmul_drain_block: int, transpose: TransposeChain
) -> _Match | None:
    """Validate one ``matmul -> drain -> transpose`` segment."""
    result: _Match | None = None
    memset_leaf = single_leaf(ir.tree, memset_block)
    matmul_leaf = single_leaf(ir.tree, matmul_block)
    drain_leaf = single_leaf(ir.tree, matmul_drain_block)
    if memset_leaf is not None and matmul_leaf is not None and drain_leaf is not None:
        memset = ir.tree.isa(memset_leaf)
        matmul = ir.tree.isa(matmul_leaf)
        drain = ir.tree.isa(drain_leaf)
        operations = memset.op_cls is NKIMemset and matmul.op_cls is NKIMatmul and drain.op_cls is NKITensorCopy
        if operations:
            old_psum = matmul.operand_bindings["dst"].tensor
            old_output = drain.operand_bindings["dst"].tensor
            connected = (
                memset.operand_bindings["dst"].tensor == old_psum
                and drain.operand_bindings["src"].tensor == old_psum
                and transpose.source == old_output
            )
            buffers = ir.all_buffers()
            names = (
                matmul.operand_bindings["stationary"].tensor,
                matmul.operand_bindings["moving"].tensor,
                old_psum,
                old_output,
                transpose.psum,
                transpose.output,
            )
            if connected and all(name in buffers for name in names):
                stationary, moving, old_psum_buffer, old_output_buffer, transpose_psum, transpose_output = (
                    buffers[name] for name in names
                )
                matmul_block_data = ir.tree.block(matmul_block)
                axes = all(axis in matmul_block_data.axis_map for axis in ("K", "M", "N"))
                rank_two = all(
                    len(buffer.shape) == 2
                    for buffer in (
                        stationary,
                        moving,
                        old_psum_buffer,
                        old_output_buffer,
                        transpose_psum,
                        transpose_output,
                    )
                )
                shapes = rank_two and (
                    stationary.shape[0] == moving.shape[0]
                    and old_psum_buffer.shape == (stationary.shape[1], moving.shape[1])
                    and old_output_buffer.shape == old_psum_buffer.shape
                    and transpose_psum.shape == old_output_buffer.shape[::-1]
                    and transpose_output.shape == old_output_buffer.shape[::-1]
                )
                storage = (
                    stationary.location == "sbuf"
                    and moving.location == "sbuf"
                    and old_psum_buffer.location == "psum"
                    and old_output_buffer.location == "sbuf"
                )
                dtype = (
                    len(
                        {
                            stationary.dtype,
                            moving.dtype,
                            old_psum_buffer.dtype,
                            old_output_buffer.dtype,
                            transpose_psum.dtype,
                            transpose_output.dtype,
                        }
                    )
                    == 1
                )
                physical_dtype = (
                    old_psum_buffer.storage_dtype == NKIMatmul.OUTPUT_STORAGE_DTYPE
                    and transpose_psum.storage_dtype == NKITranspose.OUTPUT_STORAGE_DTYPE
                    and stationary.physical_dtype() == stationary.dtype
                    and moving.physical_dtype() == moving.dtype
                )
                swapped_matmul_legal = axes and (
                    canonical_spec(
                        ir,
                        NKIMatmul,
                        {"stationary": moving.name, "moving": stationary.name, "dst": transpose_psum.name},
                        {
                            "K": matmul_block_data.axis_map["K"],
                            "M": matmul_block_data.axis_map["N"],
                            "N": matmul_block_data.axis_map["M"],
                        },
                        {},
                    )
                    is not None
                )
                exact_old_psum = set(ir.dependency.touches_by_tensor.get(old_psum, ())) == {
                    memset_leaf,
                    matmul_leaf,
                    drain_leaf,
                }
                exact_old_output = set(ir.dependency.touches_by_tensor.get(old_output, ())) == {
                    drain_leaf,
                    transpose.transpose_leaf,
                }
                if (
                    axes
                    and shapes
                    and storage
                    and dtype
                    and physical_dtype
                    and swapped_matmul_legal
                    and exact_old_psum
                    and exact_old_output
                ):
                    result = _Match(
                        memset_block=memset_block,
                        matmul_block=matmul_block,
                        matmul_drain_block=matmul_drain_block,
                        transpose=transpose,
                        stationary=stationary.name,
                        moving=moving.name,
                        old_psum=old_psum,
                        old_output=old_output,
                    )
    return result


def _apply_match(ir: KernelIR, match: _Match) -> None:
    """Swap the matmul and consume the following logical transpose."""
    matmul_block = ir.tree.block(match.matmul_block)
    k_axis = matmul_block.axis_map["K"]
    old_m_axis = matmul_block.axis_map["M"]
    old_n_axis = matmul_block.axis_map["N"]
    transpose_psum = ir.buffer(match.transpose.psum)
    replace_buffer(ir, replace(transpose_psum, storage_dtype=NKIMatmul.OUTPUT_STORAGE_DTYPE))

    memset_spec = required_spec(
        ir, NKIMemset, {"dst": match.transpose.psum}, {"P": old_n_axis, "F": old_m_axis}, {"value": 0.0}
    )
    matmul_spec = required_spec(
        ir,
        NKIMatmul,
        {"stationary": match.moving, "moving": match.stationary, "dst": match.transpose.psum},
        {"K": k_axis, "M": old_n_axis, "N": old_m_axis},
        {},
    )
    drain_spec = required_spec(
        ir,
        NKITensorCopy,
        {"src": match.transpose.psum, "dst": match.transpose.output},
        {"P": old_n_axis, "F": old_m_axis},
        {},
    )
    rewrite_block(ir.tree, match.memset_block, memset_spec)
    rewrite_block(ir.tree, match.matmul_block, matmul_spec)
    rewrite_block(ir.tree, match.matmul_drain_block, drain_spec)

    removed_blocks = [match.transpose.transpose_block, match.transpose.drain_block]
    _replace_in_parent_children(ir.tree, ir.tree.root, removed_blocks, [])
    for block_nid in removed_blocks:
        ir.tree.graph.remove_nodes_from({block_nid, *ir.tree.descendants(block_nid)})
    remove_buffers(ir, {match.old_psum, match.old_output})


__all__ = ["TransposeThroughMatmul", "TransposeThroughMatmulOption"]
