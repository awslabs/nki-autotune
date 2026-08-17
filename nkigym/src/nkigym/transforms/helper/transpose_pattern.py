"""Structural matching for one logical transpose in concrete NKI IR."""

from __future__ import annotations

from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
from nkigym.transforms.helper.canonical_rewrite import _canonical_context, is_canonical_block, single_leaf


@dataclass(frozen=True)
class TransposeChain:
    """An ``nc_transpose`` and its required PSUM-to-SBUF drain."""

    transpose_block: int
    drain_block: int
    transpose_leaf: int
    drain_leaf: int
    source: str
    psum: str
    output: str
    source_axes: tuple[str, str]


def match_transpose_chain(
    ir: KernelIR, transpose_block: int, drain_block: int, adjacent: bool | None = None
) -> TransposeChain | None:
    """Return one canonical adjacent transpose chain."""
    result: TransposeChain | None = None
    if adjacent is None:
        root_children = ir.tree.children(ir.tree.root)
        adjacent = False
        if transpose_block in root_children:
            index = root_children.index(transpose_block)
            adjacent = index + 1 < len(root_children) and root_children[index + 1] == drain_block
    if adjacent and is_canonical_block(ir, transpose_block) and is_canonical_block(ir, drain_block):
        transpose_leaf = single_leaf(ir.tree, transpose_block)
        drain_leaf = single_leaf(ir.tree, drain_block)
        if transpose_leaf is not None and drain_leaf is not None:
            transpose = ir.tree.isa(transpose_leaf)
            drain = ir.tree.isa(drain_leaf)
            if transpose.op_cls is NKITranspose and drain.op_cls is NKITensorCopy:
                source = transpose.operand_bindings["data"].tensor
                psum = transpose.operand_bindings["dst"].tensor
                output = drain.operand_bindings["dst"].tensor
                connected = drain.operand_bindings["src"].tensor == psum
                buffers = _canonical_context(ir).buffers
                names_exist = all(name in buffers for name in (source, psum, output))
                if connected and names_exist:
                    source_buffer = buffers[source]
                    psum_buffer = buffers[psum]
                    output_buffer = buffers[output]
                    shapes = (
                        len(source_buffer.shape) == 2
                        and psum_buffer.shape == source_buffer.shape[::-1]
                        and output_buffer.shape == source_buffer.shape[::-1]
                    )
                    storage = (
                        source_buffer.location == "sbuf"
                        and psum_buffer.location == "psum"
                        and output_buffer.location == "sbuf"
                    )
                    dtype = source_buffer.dtype == psum_buffer.dtype == output_buffer.dtype
                    physical_dtype = all(
                        buffer.physical_dtype() == source_buffer.dtype
                        for buffer in (source_buffer, psum_buffer, output_buffer)
                    )
                    transpose_axes = ir.tree.block(transpose_block).axis_map
                    drain_axes = ir.tree.block(drain_block).axis_map
                    axes = (
                        isinstance(transpose_axes.get("P"), str)
                        and isinstance(transpose_axes.get("F"), str)
                        and drain_axes.get("P") == transpose_axes.get("F")
                        and drain_axes.get("F") == transpose_axes.get("P")
                    )
                    exact_psum = set(ir.dependency.touches_by_tensor.get(psum, ())) == {transpose_leaf, drain_leaf}
                    if shapes and storage and dtype and physical_dtype and axes and exact_psum:
                        source_p = transpose_axes["P"]
                        source_f = transpose_axes["F"]
                        assert isinstance(source_p, str)
                        assert isinstance(source_f, str)
                        result = TransposeChain(
                            transpose_block=transpose_block,
                            drain_block=drain_block,
                            transpose_leaf=transpose_leaf,
                            drain_leaf=drain_leaf,
                            source=source,
                            psum=psum,
                            output=output,
                            source_axes=(source_p, source_f),
                        )
    return result


__all__ = ["TransposeChain", "match_transpose_chain"]
