"""Interleave a matrix product's contraction factors across native storage axes."""

from dataclasses import dataclass, replace
from typing import Any

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Expr, Var
from nkigym.ir.canonical_build import _build_subblock
from nkigym.ir.dimension_analysis import TensorDims, _AnalysisResult, _OpRecord
from nkigym.ir.program_sharding import configured_program_shards
from nkigym.ir.tree import AccessPattern, Buffer, BufferRegion, ForNode, KernelTree, partition_extent
from nkigym.ops.base import NKIOp
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.interleaved_matmul import NKIInterleavedMatmul
from nkigym.ops.interleaved_right_load import NKIInterleavedRightLoad
from nkigym.ops.interleaved_stationary_load import NKIInterleavedStationaryLoad
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.partition_pack_copy import NKIPartitionPackCopy
from nkigym.ops.store import NKIStore
from nkigym.ops.strided_tensor_copy import NKIStridedTensorCopy
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
)
from nkigym.transforms.copy_propagation import source_remains_stable
from nkigym.transforms.helper.canonical_rewrite import (
    append_root_buffers,
    axis_extents,
    block_chain,
    finalize_rewrite,
    owning_block,
    replace_buffer,
    single_leaf,
)
from nkigym.transforms.helper.normalize import _substitute_block_regions
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children


@dataclass(frozen=True)
class InterleaveContractionOption(TransformOption):
    """Select one canonical matrix-product block."""

    matmul_block_nid: int


@dataclass(frozen=True)
class _Match:
    """One product whose complete current operands can be repacked."""

    block: int
    leaf: int
    left: Buffer
    right: Buffer
    output: Buffer
    rows: int
    columns: int
    tiles: int


@dataclass(frozen=True)
class PackingFold:
    """One redundant HBM copy roundtrip feeding a packing load."""

    block: int
    source: Buffer
    output: Buffer
    rows: int
    tiles: int


class InterleaveContraction(Transform[InterleaveContractionOption]):
    """Replace one contraction layout while retaining its arithmetic and outputs.

    Native partition lanes select every ``tiles``-th contraction element, and
    the outer reduction visits the remaining elements. The current operands
    are materialized and reloaded in that representation. Replacing the HBM
    roundtrip with a SBUF permutation is a separate OnChipReshape action.
    """

    def analyze(self, ir: KernelIR) -> list[InterleaveContractionOption]:
        """Offer complete canonical products without inspecting their producers."""
        return [
            option
            for nid in ir.tree.children(ir.tree.root)
            if (leaf := single_leaf(ir.tree, nid)) is not None
            and ir.tree.isa(leaf).op_cls is NKIMatmul
            and _match(ir, option := InterleaveContractionOption(nid)) is not None
        ]

    def apply(self, ir: KernelIR, option: InterleaveContractionOption) -> KernelIR:
        """Recheck the operand geometry and rewrite a copied product."""
        if _match(ir, option) is None:
            raise TransformLegalityError(f"illegal InterleaveContraction option: {option}")
        result = copy_for_rewrite(ir)
        copied = _match(result, option)
        if copied is None:
            raise AssertionError("InterleaveContraction match disappeared after copying")
        _rewrite(result, copied)
        finalize_rewrite(result)
        return result


def _analysis(ir: KernelIR, extents: dict[str, int]) -> _AnalysisResult:
    """Supply existing buffer geometry to the canonical operand-view builder."""
    tensors = {
        name: TensorDims(name, buffer.shape, (), buffer.location, buffer.dtype, buffer.storage_dtype)
        for name, buffer in ir.all_buffers().items()
    }
    return _AnalysisResult("", [], (), extents, tensors, [], [])


def _canonical_leaf(ir: KernelIR, block: int) -> int | None:
    """Match complete canonical payloads, including every explicit tensor view."""
    leaf = single_leaf(ir.tree, block)
    if (
        leaf is None
        or ir.tree.parent(block) != ir.tree.root
        or ir.tree.block(block).alloc_buffers
        or intersects_software_pipeline(ir, (block,))
        or any(loop in ir.tree.descendants(block) for loop in configured_program_shards(ir))
    ):
        return None
    operation = ir.tree.isa(leaf)
    if any(
        buffer.location != "shared_hbm"
        and (
            buffer.partition_extent() != partition_extent(buffer.shape[0])
            or buffer.free_alignment != 1
            or buffer.versions != 1
        )
        for region in operation.operand_bindings.values()
        for buffer in (ir.buffer(region.tensor),)
    ):
        return None
    record = _OpRecord(
        operation.op_cls,
        {slot: region.tensor for slot, region in operation.operand_bindings.items()},
        ir.tree.block(block).axis_map,
        operation.kwargs,
    )
    tree = KernelTree()
    expected = _build_subblock(tree, tree.root, record, _analysis(ir, axis_extents(ir)))
    substitutions: dict[str, Expr] = {
        canonical.name: actual
        for canonical, actual in zip(tree.block(expected).iter_values, ir.tree.block(block).iter_values, strict=True)
        if isinstance(canonical, Var) and isinstance(actual, Var)
    }
    for nid in tree.preorder(expected):
        node = tree.data(nid)
        if isinstance(node, ForNode):
            renamed = substitutions.get(node.loop_var)
            if isinstance(renamed, Var):
                tree.graph.nodes[nid]["data"] = replace(node, loop_var=renamed.name)
    _substitute_block_regions(tree, expected, substitutions)
    return leaf if block_chain(ir.tree, block) == block_chain(tree, expected) else None


def _producer(ir: KernelIR, tensor: str) -> int | None:
    """Return the sole complete canonical writer of one tensor."""
    writers = [
        nid for nid in ir.dependency.touches_by_tensor.get(tensor, ()) if tensor in ir.dependency.info(nid).writes
    ]
    return (
        writers[0]
        if len(writers) == 1 and _canonical_leaf(ir, owning_block(ir.tree, writers[0])) == writers[0]
        else None
    )


def _full_region(buffer: Buffer) -> BufferRegion:
    """Return the complete logical tensor footprint."""
    return BufferRegion(
        tensor=buffer.name, ranges=tuple((Const(value=0), Const(value=extent)) for extent in buffer.shape)
    )


def _same_storage(buffers: tuple[Buffer, ...]) -> bool:
    """Require matching logical and native floating-point storage."""
    first = buffers[0]
    return first.physical_dtype() in {"float32", "bfloat16", "float16"} and all(
        buffer.dtype == first.dtype and buffer.physical_dtype() == first.physical_dtype() and buffer.versions == 1
        for buffer in buffers
    )


def match_packing_fold(ir: KernelIR, block: int, previous: int | None) -> PackingFold | None:
    """Prove the current SBUF matrix still holds the stored reshape input."""
    if previous is None:
        return None
    load_nid, store_nid = single_leaf(ir.tree, block), single_leaf(ir.tree, previous)
    if load_nid is None or store_nid is None:
        return None
    load, store = ir.tree.isa(load_nid), ir.tree.isa(store_nid)
    if (
        load.op_cls not in {NKIInterleavedStationaryLoad, NKIInterleavedRightLoad}
        or store.op_cls is not NKIStore
        or store.kwargs
        or load.operand_bindings["src"].tensor != store.operand_bindings["dst"].tensor
    ):
        return None
    if _canonical_leaf(ir, block) is None or _canonical_leaf(ir, previous) is None:
        return None
    tiles = load.kwargs.get("tiles")
    if not isinstance(tiles, int) or tiles < 1 or load.kwargs != {"tiles": tiles}:
        return None
    current = ir.buffer(store.operand_bindings["src"].tensor)
    stored = ir.buffer(store.operand_bindings["dst"].tensor)
    output = ir.buffer(load.operand_bindings["dst"].tensor)
    if (
        len(current.shape) != 2
        or current.physical_dtype() not in {"float32", "bfloat16", "float16"}
        or stored.shape != current.shape
        or stored.location != "shared_hbm"
        or _producer(ir, stored.name) != store_nid
        or not _same_storage((current, stored, output))
    ):
        return None
    rows = current.shape[1]
    transpose_tile = NKIDMATranspose.MIN_TILE_SIZE["F"]
    if tiles > 1 and rows > transpose_tile and rows % transpose_tile:
        return None
    positions = {nid: index for index, nid in enumerate(ir.tree.preorder())}
    if current.location != "sbuf" or not source_remains_stable(
        ir, _full_region(current), store_nid, load_nid, positions
    ):
        return None
    return PackingFold(block, current, output, rows, tiles)


def _match(ir: KernelIR, option: InterleaveContractionOption) -> _Match | None:
    """Require complete inputs, live sources, and an unmodified additive product."""
    if option.matmul_block_nid not in ir.tree.children(ir.tree.root):
        return None
    leaf_nid = _canonical_leaf(ir, option.matmul_block_nid)
    if leaf_nid is None:
        return None
    leaf = ir.tree.isa(leaf_nid)
    if (
        leaf.op_cls is not NKIMatmul
        or set(leaf.operand_bindings) != {"stationary", "moving", "dst"}
        or set(leaf.kwargs) - {"name", "accumulate"}
        or leaf.kwargs.get("accumulate", True) is not True
    ):
        return None
    stationary, moving, output = (
        ir.buffer(leaf.operand_bindings[slot].tensor) for slot in ("stationary", "moving", "dst")
    )
    if (
        any(len(buffer.shape) != 2 for buffer in (stationary, moving, output))
        or stationary.shape[0] <= 128
        or stationary.shape[0] % 128
        or moving.shape[0] != stationary.shape[0]
        or output.shape != (stationary.shape[1], moving.shape[1])
        or stationary.location != "sbuf"
        or moving.location != "sbuf"
        or output.location != "psum"
        or output.physical_dtype() != "float32"
        or not _same_storage((stationary, moving))
        or any(buffer.free_alignment != 1 for buffer in (stationary, moving))
    ):
        return None
    rows, columns, tiles = stationary.shape[1], moving.shape[1], stationary.shape[0] // 128
    return _Match(option.matmul_block_nid, leaf_nid, stationary, moving, output, rows, columns, tiles)


class _PackingBuilder:
    """Build explicit representation operations without scheduling them."""

    def __init__(self, ir: KernelIR, shared_axes: dict[str, str]) -> None:
        """Retain the existing axis identities needed by a rewritten consumer."""
        self.ir = ir
        self.names = set(ir.all_buffers())
        self.extents = axis_extents(ir)
        self.serial = 0
        self.blocks: list[int] = []
        self.shared_axes = shared_axes

    def temporary(self, stem: str, shape: tuple[int, ...], source: Buffer, location: str) -> Buffer:
        """Declare one fresh representation buffer at the root."""
        name = stem
        while name in self.names:
            name += "_"
        self.names.add(name)
        buffer = Buffer(
            name=name,
            shape=shape,
            dtype=source.dtype,
            storage_dtype=source.storage_dtype,
            location=location,
            partition_size=partition_extent(shape[0]) if location == "sbuf" else None,
        )
        append_root_buffers(self.ir, (buffer,))
        return buffer

    def append(
        self, op_cls: type[NKIOp], operands: dict[str, Buffer], sizes: dict[str, int], kwargs: dict[str, Any]
    ) -> int:
        """Append one native operation, sharing only its declared packing axes."""
        mapping = {}
        shared = op_cls in {NKIInterleavedRightLoad, NKIInterleavedStationaryLoad, NKIInterleavedMatmul}
        for abstract, extent in sizes.items():
            if shared and abstract in self.shared_axes:
                mapping[abstract] = self.shared_axes[abstract]
                continue
            while f"d{self.serial}" in self.extents:
                self.serial += 1
            concrete = f"d{self.serial}"
            mapping[abstract] = concrete
            self.extents[concrete] = extent
            if shared:
                self.shared_axes[abstract] = concrete
            self.serial += 1
        record = _OpRecord(op_cls, {slot: buffer.name for slot, buffer in operands.items()}, mapping, kwargs)
        block = _build_subblock(self.ir.tree, self.ir.tree.root, record, _analysis(self.ir, self.extents))
        self.blocks.append(block)
        return block

    def replace(self, block: int) -> None:
        """Replace one old operation with the emitted representation sequence."""
        for created in self.blocks:
            self.ir.tree.graph.remove_edge(self.ir.tree.root, created)
        _replace_in_parent_children(self.ir.tree, self.ir.tree.root, [block], self.blocks)
        self.ir.tree.graph.remove_nodes_from((block, *self.ir.tree.descendants(block)))


def _rewrite(ir: KernelIR, match: _Match) -> None:
    """Repack the current operands without forwarding through their producers."""
    builder = _PackingBuilder(ir, {axis: ir.tree.block(match.block).axis_map[axis] for axis in ("M", "N")})
    left = builder.temporary(f"{match.output.name}_left", (128, match.rows * match.tiles), match.left, "sbuf")
    right = builder.temporary(f"{match.output.name}_right", (128, match.tiles * match.columns), match.right, "sbuf")
    for source, output, op_cls, free_axis, free in (
        (match.left, left, NKIInterleavedStationaryLoad, "M", match.rows),
        (match.right, right, NKIInterleavedRightLoad, "N", match.columns),
    ):
        stored = builder.temporary(f"{output.name}_stored", source.shape, source, "shared_hbm")
        builder.append(NKIStore, {"src": source, "dst": stored}, {"P": match.tiles * 128, "F": free}, {})
        builder.append(
            op_cls,
            {"src": stored, "dst": output},
            {"P": 128, "K": match.tiles, free_axis: free},
            {"tiles": match.tiles},
        )
    old = ir.tree.isa(match.leaf)
    sizes = {"K": match.tiles, "P": 128, "M": match.rows, "N": match.columns}
    order = tuple(
        rewritten
        for axis in ir.tree.block(match.block).axis_map
        for rewritten in (("K", "P") if axis == "K" else (axis,))
    )
    builder.append(
        NKIInterleavedMatmul,
        {"stationary": left, "moving": right, "dst": match.output},
        {axis: sizes[axis] for axis in order},
        {**old.kwargs, "tiles": match.tiles, "accumulate": True},
    )
    builder.replace(match.block)


def fold_packing(ir: KernelIR, match: PackingFold) -> None:
    """Read the current SBUF matrix without forwarding through its producer."""
    builder = _PackingBuilder(ir, dict(ir.tree.block(match.block).axis_map))
    source, output = match.source, match.output
    if match.tiles == 1:
        builder.append(NKILoad, {"src": source, "dst": output}, {"P": 128, "F": match.rows}, {})
        builder.replace(match.block)
        return
    first = builder.temporary(f"{output.name}_transposed", (match.rows, 128 * match.tiles), source, "sbuf")
    second = builder.temporary(f"{output.name}_reordered", (match.rows, 128 * match.tiles), source, "sbuf")
    builder.append(NKIDMATranspose, {"src": source, "dst": first}, {"P": 128 * match.tiles, "F": match.rows}, {})
    builder.append(
        NKIStridedTensorCopy,
        {"src": first, "dst": second},
        {"P": match.rows, "F": 128 * match.tiles, "N": 128 * match.tiles},
        {
            "pattern": ((1, match.tiles), (match.tiles, 128)),
            "offset": 0,
            "width": 128 * match.tiles,
            "engine": "vector",
        },
    )
    third = builder.temporary(f"{output.name}_partition_tiles", (128 * match.tiles, match.rows), source, "sbuf")
    third = replace(third, free_alignment=8 if third.physical_dtype() == "float32" else 16)
    replace_buffer(ir, third)
    builder.append(NKIDMATranspose, {"src": second, "dst": third}, {"P": match.rows, "F": 128 * match.tiles}, {})
    packed_block = builder.append(
        NKIPartitionPackCopy,
        {"src": third, "dst": output},
        {"P": 128, "K": match.tiles, "F": match.rows},
        {"tiles": match.tiles, "engine": "vector"},
    )
    leaf_nid = single_leaf(ir.tree, packed_block)
    if leaf_nid is None:
        raise AssertionError("partition pack copy has no unique instruction")
    leaf = ir.tree.isa(leaf_nid)
    pitch = third.per_tile_physical_shape()[2]
    source_pattern = AccessPattern(
        pattern=(
            (Const(value=match.tiles * pitch), Const(value=128)),
            (Const(value=pitch), Const(value=match.tiles)),
            (Const(value=1), Const(value=match.rows)),
        ),
        offset=Const(value=0),
    )
    ir.tree.graph.nodes[leaf_nid]["data"] = replace(
        leaf, access_patterns={**leaf.access_patterns, "src": source_pattern}
    )
    builder.replace(match.block)


__all__ = ["InterleaveContraction", "InterleaveContractionOption"]
