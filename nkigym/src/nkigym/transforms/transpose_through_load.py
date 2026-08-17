"""Commute one logical transpose upward through an HBM-to-SBUF load."""

from __future__ import annotations

from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Mul, Var
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, IterVar
from nkigym.ops.base import AxisRole
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.search.state_facts import operation_facts
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite
from nkigym.transforms.helper.canonical_rewrite import finalize_rewrite, is_canonical_block, single_leaf


@dataclass(frozen=True)
class TransposeThroughLoadOption(TransformOption):
    """Commute one transpose through its load."""

    target_nid: int


@dataclass(frozen=True)
class _Match:
    """One canonical load followed by an SBUF DMA transpose."""

    load_block: int
    transpose_block: int
    source: str
    loaded: str
    output: str
    first_axis: str
    second_axis: str
    first_tile: int
    second_tile: int


class TransposeThroughLoad(Transform[TransposeThroughLoadOption]):
    """Apply ``T(Load(x)) = DMATranspose(x)``."""

    def analyze(self, ir: KernelIR) -> list[TransposeThroughLoadOption]:
        """Return every eligible canonical load block."""
        if not operation_facts(ir).has_ops(NKILoad, NKIDMATranspose):
            return []
        options: list[TransposeThroughLoadOption] = []
        root_children = tuple(ir.tree.children(ir.tree.root))
        for index, block_nid in enumerate(root_children):
            leaf_nid = single_leaf(ir.tree, block_nid)
            if leaf_nid is None or ir.tree.isa(leaf_nid).op_cls is not NKILoad:
                continue
            match = _match(ir, block_nid, root_children, index)
            if match is not None:
                options.append(TransposeThroughLoadOption(target_nid=block_nid))
        return options

    def apply(self, ir: KernelIR, option: TransposeThroughLoadOption) -> KernelIR:
        """Recheck and commute one materialized DMA transpose into its load."""
        match = _match(ir, option.target_nid)
        if match is None:
            raise TransformLegalityError(
                f"TransposeThroughLoad target {option.target_nid} is not an eligible canonical load-transpose segment"
            )
        new_ir = copy_for_rewrite(ir)
        copied_match = _match(new_ir, option.target_nid)
        if copied_match is None:
            raise AssertionError("TransposeThroughLoad match disappeared after deepcopy")
        _apply_match(new_ir, copied_match)
        finalize_rewrite(new_ir)
        return new_ir


def _match(
    ir: KernelIR, target_nid: int, root_children: tuple[int, ...] | None = None, index: int | None = None
) -> _Match | None:
    """Return a legal segment beginning at ``target_nid``."""
    result: _Match | None = None
    children = tuple(ir.tree.children(ir.tree.root)) if root_children is None else root_children
    if index is None and target_nid in children:
        index = children.index(target_nid)
    if index is not None:
        if index + 1 < len(children):
            transpose_block = children[index + 1]
            result = _validate(ir, target_nid, transpose_block)
    return result


def _validate(ir: KernelIR, load_block: int, transpose_block: int) -> _Match | None:
    """Validate one adjacent canonical load-DMA-transpose chain."""
    result: _Match | None = None
    load_leaf = single_leaf(ir.tree, load_block)
    transpose_leaf = single_leaf(ir.tree, transpose_block)
    leaves_exist = load_leaf is not None and transpose_leaf is not None
    if leaves_exist:
        assert load_leaf is not None
        assert transpose_leaf is not None
        load = ir.tree.isa(load_leaf)
        transpose = ir.tree.isa(transpose_leaf)
        op_chain = load.op_cls is NKILoad and transpose.op_cls is NKIDMATranspose
        canonical = all(is_canonical_block(ir, block) for block in (load_block, transpose_block))
        if op_chain and canonical:
            source = load.operand_bindings["src"].tensor
            loaded = load.operand_bindings["dst"].tensor
            output = transpose.operand_bindings["dst"].tensor
            connected = transpose.operand_bindings["src"].tensor == loaded
            buffers = ir.all_buffers()
            names_exist = all(name in buffers for name in (source, loaded, output))
            if connected and names_exist:
                source_buffer = buffers[source]
                loaded_buffer = buffers[loaded]
                output_buffer = buffers[output]
                shapes = (
                    len(source_buffer.shape) == 2
                    and loaded_buffer.shape == source_buffer.shape
                    and output_buffer.shape == source_buffer.shape[::-1]
                )
                storage = (
                    source_buffer.location == "shared_hbm"
                    and source in ir.param_buffers
                    and loaded_buffer.location == "sbuf"
                    and output_buffer.location == "sbuf"
                )
                dtype = len({source_buffer.dtype, loaded_buffer.dtype, output_buffer.dtype}) == 1
                physical_dtype = all(
                    buffer.physical_dtype() == source_buffer.dtype
                    for buffer in (source_buffer, loaded_buffer, output_buffer)
                )
                transpose_source = transpose.operand_bindings["src"]
                tiles = tuple(
                    width.value if isinstance(width, Const) and isinstance(width.value, int) else None
                    for _lower, width in transpose_source.ranges
                )
                tileable = len(tiles) == 2 and all(
                    tile is not None
                    and NKIDMATranspose.MIN_TILE_SIZE[axis] <= tile
                    and tile <= NKIDMATranspose.HBM_SOURCE_MAX_TILE_SIZE[axis]
                    and extent % tile == 0
                    for extent, axis, tile in zip(source_buffer.shape, ("P", "F"), tiles)
                )
                exact_loaded = set(ir.dependency.touches_by_tensor.get(loaded, ())) == {load_leaf, transpose_leaf}
                deleted_allocations = {buffer.name for buffer in ir.tree.block(transpose_block).alloc_buffers}
                preserves_ownership = deleted_allocations <= {loaded, output}
                axis_map = ir.tree.block(load_block).axis_map
                first_axis = axis_map.get("P")
                second_axis = axis_map.get("F")
                axes = isinstance(first_axis, str) and isinstance(second_axis, str)
                if (
                    shapes
                    and storage
                    and dtype
                    and physical_dtype
                    and tileable
                    and exact_loaded
                    and preserves_ownership
                    and axes
                ):
                    assert isinstance(first_axis, str)
                    assert isinstance(second_axis, str)
                    first_tile, second_tile = tiles
                    assert isinstance(first_tile, int)
                    assert isinstance(second_tile, int)
                    result = _Match(
                        load_block=load_block,
                        transpose_block=transpose_block,
                        source=source,
                        loaded=loaded,
                        output=output,
                        first_axis=first_axis,
                        second_axis=second_axis,
                        first_tile=first_tile,
                        second_tile=second_tile,
                    )
    return result


def _apply_match(ir: KernelIR, match: _Match) -> None:
    """Rewrite a validated segment in place."""
    source = ir.buffer(match.source)
    first_tile = match.first_tile
    second_tile = match.second_tile
    first_trip = source.shape[0] // first_tile
    second_trip = source.shape[1] // second_tile
    first_loop = f"i_{match.first_axis}_0"
    second_loop = f"i_{match.second_axis}_0"
    first_value = Var(name=first_loop) if first_trip > 1 else Const(value=0)
    second_value = Var(name=second_loop) if second_trip > 1 else Const(value=0)
    src_region = BufferRegion(
        tensor=match.source,
        ranges=(
            (Mul(left=first_value, right=Const(value=first_tile)), Const(value=first_tile)),
            (Mul(left=second_value, right=Const(value=second_tile)), Const(value=second_tile)),
        ),
    )
    dst_region = BufferRegion(
        tensor=match.output,
        ranges=(
            (second_value, Const(value=second_tile)),
            (Mul(left=first_value, right=Const(value=first_tile)), Const(value=first_tile)),
        ),
    )
    allocations = ir.tree.block(match.transpose_block).alloc_buffers
    block = BlockNode(
        iter_vars=(
            IterVar(axis=match.first_axis, dom=(0, source.shape[0]), role=AxisRole.PARALLEL),
            IterVar(axis=match.second_axis, dom=(0, source.shape[1]), role=AxisRole.PARALLEL),
        ),
        iter_values=(first_value, second_value),
        reads=(src_region,),
        writes=(dst_region,),
        alloc_buffers=allocations,
        axis_map={"P": match.first_axis, "F": match.second_axis},
    )
    descendants = list(ir.tree.descendants(match.transpose_block))
    ir.tree.graph.remove_nodes_from(descendants)
    ir.tree.graph.nodes[match.transpose_block]["data"] = block
    parent = match.transpose_block
    if first_trip > 1:
        parent = ir.tree.add_node(ForNode(loop_var=first_loop, extent=first_trip), parent=parent)
    if second_trip > 1:
        parent = ir.tree.add_node(ForNode(loop_var=second_loop, extent=second_trip), parent=parent)
    ir.tree.add_node(
        ISANode(op_cls=NKIDMATranspose, operand_bindings={"src": src_region, "dst": dst_region}, kwargs={}),
        parent=parent,
    )


__all__ = ["TransposeThroughLoad", "TransposeThroughLoadOption"]
