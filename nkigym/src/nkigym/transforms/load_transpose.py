"""Fuse a load, Tensor Engine transpose, and drain into DMA transpose."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Mul, Var
from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, IterVar
from nkigym.ops.base import AxisRole
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class LoadTransposeOption(TransformOption):
    """Fuse the canonical load block at ``target_nid``."""

    target_nid: int


@dataclass(frozen=True)
class _Match:
    """One canonical load, transpose, and drain segment."""

    load_block: int
    transpose_block: int
    drain_block: int
    source: str
    loaded: str
    transpose_psum: str
    output: str
    first_axis: str
    second_axis: str


class LoadTranspose(Transform[LoadTransposeOption]):
    """Replace ``load -> nc_transpose -> drain`` with tiled DMA transpose."""

    def analyze(self, ir: KernelIR) -> list[LoadTransposeOption]:
        """Return every eligible canonical load block."""
        options: list[LoadTransposeOption] = []
        for block_nid in ir.tree.children(ir.tree.root):
            if _match(ir, block_nid) is not None:
                options.append(LoadTransposeOption(target_nid=block_nid))
        return options

    def apply(self, ir: KernelIR, option: LoadTransposeOption) -> KernelIR:
        """Recheck and atomically fuse one canonical segment."""
        match = _match(ir, option.target_nid)
        if match is None:
            raise TransformLegalityError(
                f"LoadTranspose target {option.target_nid} is not an eligible canonical load-transpose segment"
            )
        new_ir = copy.deepcopy(ir)
        copied_match = _match(new_ir, option.target_nid)
        if copied_match is None:
            raise AssertionError("LoadTranspose match disappeared after deepcopy")
        _apply_match(new_ir, copied_match)
        place_buffers(new_ir.tree)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir


def _single_leaf(ir: KernelIR, block_nid: int) -> int | None:
    """Return the sole ISA leaf below ``block_nid``."""
    leaves = [nid for nid in ir.tree.descendants(block_nid) if isinstance(ir.tree.data(nid), ISANode)]
    result = leaves[0] if len(leaves) == 1 else None
    return result


def _canonical_chain(ir: KernelIR, block_nid: int) -> bool:
    """Whether a block is one unbranched loop chain ending in one ISA."""
    current = block_nid
    valid = True
    complete = False
    while valid and not complete:
        children = ir.tree.children(current)
        valid = len(children) == 1
        if valid:
            current = children[0]
            payload = ir.tree.data(current)
            valid = not isinstance(payload, BlockNode)
            complete = isinstance(payload, ISANode)
    return valid and complete


def _match(ir: KernelIR, target_nid: int) -> _Match | None:
    """Return a legal segment beginning at ``target_nid``."""
    result: _Match | None = None
    root_children = ir.tree.children(ir.tree.root)
    if target_nid in root_children:
        index = root_children.index(target_nid)
        if index + 2 < len(root_children):
            transpose_block = root_children[index + 1]
            drain_block = root_children[index + 2]
            result = _validate(ir, target_nid, transpose_block, drain_block)
    return result


def _validate(ir: KernelIR, load_block: int, transpose_block: int, drain_block: int) -> _Match | None:
    """Validate one adjacent canonical load-transpose-drain chain."""
    result: _Match | None = None
    load_leaf = _single_leaf(ir, load_block)
    transpose_leaf = _single_leaf(ir, transpose_block)
    drain_leaf = _single_leaf(ir, drain_block)
    leaves_exist = load_leaf is not None and transpose_leaf is not None and drain_leaf is not None
    if leaves_exist:
        assert load_leaf is not None
        assert transpose_leaf is not None
        assert drain_leaf is not None
        load = ir.tree.isa(load_leaf)
        transpose = ir.tree.isa(transpose_leaf)
        drain = ir.tree.isa(drain_leaf)
        op_chain = load.op_cls is NKILoad and transpose.op_cls is NKITranspose and drain.op_cls is NKITensorCopy
        canonical = all(_canonical_chain(ir, block) for block in (load_block, transpose_block, drain_block))
        if op_chain and canonical:
            source = load.operand_bindings["src"].tensor
            loaded = load.operand_bindings["dst"].tensor
            transpose_psum = transpose.operand_bindings["dst"].tensor
            output = drain.operand_bindings["dst"].tensor
            connected = (
                transpose.operand_bindings["data"].tensor == loaded
                and drain.operand_bindings["src"].tensor == transpose_psum
            )
            buffers = ir.all_buffers()
            names_exist = all(name in buffers for name in (source, loaded, transpose_psum, output))
            if connected and names_exist:
                source_buffer = buffers[source]
                loaded_buffer = buffers[loaded]
                psum_buffer = buffers[transpose_psum]
                output_buffer = buffers[output]
                shapes = (
                    len(source_buffer.shape) == 2
                    and loaded_buffer.shape == source_buffer.shape
                    and psum_buffer.shape == source_buffer.shape[::-1]
                    and output_buffer.shape == source_buffer.shape[::-1]
                )
                storage = (
                    source_buffer.location == "shared_hbm"
                    and loaded_buffer.location == "sbuf"
                    and psum_buffer.location == "psum"
                    and output_buffer.location == "sbuf"
                )
                dtype = len({source_buffer.dtype, loaded_buffer.dtype, psum_buffer.dtype, output_buffer.dtype}) == 1
                divisible = all(extent % 128 == 0 for extent in source_buffer.shape)
                exact_loaded = set(ir.dependency.touches_by_tensor.get(loaded, ())) == {load_leaf, transpose_leaf}
                exact_psum = set(ir.dependency.touches_by_tensor.get(transpose_psum, ())) == {
                    transpose_leaf,
                    drain_leaf,
                }
                axis_map = ir.tree.block(load_block).axis_map
                first_axis = axis_map.get("P")
                second_axis = axis_map.get("F")
                axes = isinstance(first_axis, str) and isinstance(second_axis, str)
                if shapes and storage and dtype and divisible and exact_loaded and exact_psum and axes:
                    assert isinstance(first_axis, str)
                    assert isinstance(second_axis, str)
                    result = _Match(
                        load_block=load_block,
                        transpose_block=transpose_block,
                        drain_block=drain_block,
                        source=source,
                        loaded=loaded,
                        transpose_psum=transpose_psum,
                        output=output,
                        first_axis=first_axis,
                        second_axis=second_axis,
                    )
    return result


def _apply_match(ir: KernelIR, match: _Match) -> None:
    """Rewrite a validated segment in place."""
    source = ir.buffer(match.source)
    first_tile = min(source.shape[0], 512)
    if source.shape[0] % first_tile != 0:
        raise AssertionError(f"{match.source} extent {source.shape[0]} is not tiled by {first_tile}")
    first_trip = source.shape[0] // first_tile
    second_trip = source.shape[1] // 128
    first_loop = f"i_{match.first_axis}_0"
    second_loop = f"i_{match.second_axis}_0"
    first_value = Var(name=first_loop) if first_trip > 1 else Const(value=0)
    second_value = Var(name=second_loop) if second_trip > 1 else Const(value=0)
    src_region = BufferRegion(
        tensor=match.source,
        ranges=(
            (Mul(left=first_value, right=Const(value=first_tile)), Const(value=first_tile)),
            (Mul(left=second_value, right=Const(value=128)), Const(value=128)),
        ),
    )
    dst_region = BufferRegion(
        tensor=match.output,
        ranges=(
            (second_value, Const(value=128)),
            (Mul(left=first_value, right=Const(value=first_tile)), Const(value=first_tile)),
        ),
    )
    block = BlockNode(
        iter_vars=(
            IterVar(axis=match.first_axis, dom=(0, source.shape[0]), role=AxisRole.PARALLEL),
            IterVar(axis=match.second_axis, dom=(0, source.shape[1]), role=AxisRole.PARALLEL),
        ),
        iter_values=(first_value, second_value),
        reads=(src_region,),
        writes=(dst_region,),
        alloc_buffers=(),
        axis_map={"P": match.first_axis, "F": match.second_axis},
    )
    descendants = list(ir.tree.descendants(match.load_block))
    ir.tree.graph.remove_nodes_from(descendants)
    ir.tree.graph.nodes[match.load_block]["data"] = block
    parent = match.load_block
    if first_trip > 1:
        parent = ir.tree.add_node(ForNode(loop_var=first_loop, extent=first_trip), parent=parent)
    if second_trip > 1:
        parent = ir.tree.add_node(ForNode(loop_var=second_loop, extent=second_trip), parent=parent)
    ir.tree.add_node(
        ISANode(op_cls=NKIDMATranspose, operand_bindings={"src": src_region, "dst": dst_region}, kwargs={}),
        parent=parent,
    )
    _replace_in_parent_children(
        ir.tree, ir.tree.root, [match.load_block, match.transpose_block, match.drain_block], [match.load_block]
    )
    ir.tree.graph.remove_nodes_from(
        [
            match.transpose_block,
            *ir.tree.descendants(match.transpose_block),
            match.drain_block,
            *ir.tree.descendants(match.drain_block),
        ]
    )
    _remove_buffers(ir, {match.loaded, match.transpose_psum})


def _remove_buffers(ir: KernelIR, names: set[str]) -> None:
    """Remove temporary declarations named by ``names``."""
    removed: set[str] = set()
    for block_nid in ir.tree.blocks():
        block = ir.tree.block(block_nid)
        retained = tuple(buffer for buffer in block.alloc_buffers if buffer.name not in names)
        removed.update(buffer.name for buffer in block.alloc_buffers if buffer.name in names)
        if retained != block.alloc_buffers:
            ir.tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=retained)
    if removed != names:
        raise AssertionError(f"expected to remove buffers {names}, removed {removed}")


__all__ = ["LoadTranspose", "LoadTransposeOption"]
