"""Replace one reshape load with an equivalent on-chip permutation."""

from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Mul, Var
from nkigym.ir.program_sharding import configured_program_shards
from nkigym.ir.tree import AccessPattern, BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar
from nkigym.ops.base import AxisRole
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.grouped_store import NKIGroupedStore
from nkigym.ops.load import NKILoad
from nkigym.ops.partition_unpack_copy import NKIPartitionUnpackCopy
from nkigym.ops.strided_tensor_copy import NKIStridedTensorCopy
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
)
from nkigym.transforms.helper.canonical_rewrite import (
    append_root_buffers,
    axis_extents,
    finalize_rewrite,
    is_canonical_block,
    single_leaf,
)
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children
from nkigym.transforms.interleave_contraction import fold_packing, match_packing_fold


@dataclass(frozen=True)
class OnChipReshapeOption(TransformOption):
    """Select one load immediately following a grouped reshape store."""

    load_block_nid: int


@dataclass(frozen=True)
class _Reshape:
    """One complete stored view and its adjacent on-chip destination."""

    load_block: int
    load_leaf: int
    source: Buffer
    output: Buffer
    rows: int
    parts: int
    width: int


class OnChipReshape(Transform[OnChipReshapeOption]):
    """Read a reshape directly from SBUF through exact native permutations.

    The existing store remains in place. Removing it after its result becomes
    dead is a separate EliminateDeadProducer action. Existing allocation
    layouts, arithmetic, and other consumers remain unchanged.
    """

    def analyze(self, ir: KernelIR) -> list[OnChipReshapeOption]:
        """Offer complete adjacent reshape loads with compatible SBUF geometry."""
        children = ir.tree.children(ir.tree.root)
        previous = dict(zip(children[1:], children))
        return [
            option
            for nid in previous
            if _match(ir, option := OnChipReshapeOption(nid), previous) is not None
            or _match_transpose(ir, option, previous) is not None
            or match_packing_fold(ir, nid, previous[nid]) is not None
        ]

    def apply(self, ir: KernelIR, option: OnChipReshapeOption) -> KernelIR:
        """Recheck and replace the selected load with one on-chip reshape."""
        match = _match(ir, option)
        transpose = _match_transpose(ir, option)
        children = ir.tree.children(ir.tree.root)
        previous = dict(zip(children[1:], children)).get(option.load_block_nid)
        packing = match_packing_fold(ir, option.load_block_nid, previous) if match is None else None
        if match is None and transpose is None and packing is None:
            raise TransformLegalityError(f"illegal OnChipReshape option: {option}")
        result = copy_for_rewrite(ir)
        if match is not None:
            _rewrite(result, match)
        elif transpose is not None:
            _rewrite_transpose(result, transpose)
        else:
            copied = match_packing_fold(result, option.load_block_nid, previous)
            if copied is None:
                raise AssertionError("packing fold disappeared after copying")
            fold_packing(result, copied)
        finalize_rewrite(result)
        return result


def _plain_leaf(ir: KernelIR, block_nid: int) -> int | None:
    """Require one unconditional instruction without local loops or allocations."""
    node = ir.tree.data(block_nid)
    leaf = single_leaf(ir.tree, block_nid) if isinstance(node, BlockNode) else None
    return (
        leaf
        if isinstance(node, BlockNode)
        and leaf is not None
        and ir.tree.children(block_nid) == [leaf]
        and not node.annotations
        and not node.alloc_buffers
        and all(variable.role is AxisRole.PARALLEL for variable in node.iter_vars)
        and all(value == Const(value=0) for value in node.iter_values)
        else None
    )


def _full_region(buffer: Buffer) -> BufferRegion:
    """Return one complete logical tensor footprint."""
    return BufferRegion(
        tensor=buffer.name, ranges=tuple((Const(value=0), Const(value=extent)) for extent in buffer.shape)
    )


def _match(
    ir: KernelIR, option: OnChipReshapeOption, previous_by_load: dict[int, int] | None = None
) -> _Reshape | None:
    """Prove that the preceding store supplies exactly this load's source value."""
    if previous_by_load is None:
        children = ir.tree.children(ir.tree.root)
        previous_by_load = dict(zip(children[1:], children))
    previous = previous_by_load.get(option.load_block_nid)
    if previous is None:
        return None
    store_leaf, load_leaf = _plain_leaf(ir, previous), _plain_leaf(ir, option.load_block_nid)
    if store_leaf is None or load_leaf is None:
        return None
    store, load = ir.tree.isa(store_leaf), ir.tree.isa(load_leaf)
    if (
        store.op_cls is not NKIGroupedStore
        or load.op_cls is not NKILoad
        or set(store.operand_bindings) != {"src", "dst"}
        or set(load.operand_bindings) != {"src", "dst"}
        or load.kwargs
        or load.access_patterns
        or set(store.kwargs) != {"groups", "rows", "stages"}
        or store.kwargs["groups"] != 1
        or intersects_software_pipeline(ir, (previous, option.load_block_nid))
    ):
        return None
    rows, parts = store.kwargs["rows"], store.kwargs["stages"]
    if not isinstance(rows, int) or not isinstance(parts, int) or rows < 1 or parts < 2:
        return None
    source = ir.buffer(store.operand_bindings["src"].tensor)
    stored = ir.buffer(store.operand_bindings["dst"].tensor)
    output = ir.buffer(load.operand_bindings["dst"].tensor)
    if (
        len(source.shape) != 2
        or source.shape[0] != rows * parts
        or not 1 <= source.shape[1] <= 128
        or rows * parts > 128
        or stored.shape != (rows, parts * source.shape[1])
        or output.shape != stored.shape
        or stored.location != "shared_hbm"
        or any(buffer.dtype != "float32" or buffer.physical_dtype() != "float32" for buffer in (source, stored, output))
        or source.shape[1] % (NKIDMATranspose.OUTPUT_TILE_ALIGNMENT_BYTES["dst"] // 4)
        or any(
            buffer.location != "sbuf"
            or buffer.versions != 1
            or buffer.list_len != 1
            or buffer.partition_extent() != buffer.shape[0]
            for buffer in (source, output)
        )
        or source.name == output.name
        or store.operand_bindings != {"src": _full_region(source), "dst": _full_region(stored)}
        or load.operand_bindings != {"src": _full_region(stored), "dst": _full_region(output)}
        or not _store_patterns(store, source)
    ):
        return None
    writers = [
        nid for nid in ir.dependency.touches_by_tensor[stored.name] if stored.name in ir.dependency.info(nid).writes
    ]
    if writers != [store_leaf]:
        return None
    return _Reshape(option.load_block_nid, load_leaf, source, output, rows, parts, source.shape[1])


def _store_patterns(store: ISANode, source: Buffer) -> bool:
    """Require the canonical linear grouped-store views, including source pitch."""
    shape = source.per_tile_physical_shape()
    source_pattern = AccessPattern(
        pattern=(
            (Const(value=shape[1] * shape[2]), Const(value=source.shape[0])),
            (Const(value=1), Const(value=source.shape[1])),
        ),
        offset=Const(value=0),
    )
    stored_pattern = AccessPattern(
        pattern=(
            (Const(value=source.shape[1]), Const(value=source.shape[0])),
            (Const(value=1), Const(value=source.shape[1])),
        ),
        offset=Const(value=0),
    )
    return store.access_patterns == {"src": source_pattern, "dst": stored_pattern}


def _match_transpose(
    ir: KernelIR, option: OnChipReshapeOption, previous_by_load: dict[int, int] | None = None
) -> _Reshape | None:
    """Prove a complete packed store supplies the adjacent transposed load."""
    if previous_by_load is None:
        children = ir.tree.children(ir.tree.root)
        previous_by_load = dict(zip(children[1:], children))
    previous = previous_by_load.get(option.load_block_nid)
    if previous is None:
        return None
    store_leaf = _plain_leaf(ir, previous)
    load_leaf = single_leaf(ir.tree, option.load_block_nid)
    if store_leaf is None or load_leaf is None:
        return None
    store, load = ir.tree.isa(store_leaf), ir.tree.isa(load_leaf)
    if (
        store.op_cls is not NKIGroupedStore
        or load.op_cls is not NKIDMATranspose
        or set(store.operand_bindings) != {"src", "dst"}
        or set(load.operand_bindings) != {"src", "dst"}
        or load.kwargs
        or load.access_patterns
        or set(store.kwargs) != {"groups", "rows", "stages"}
        or store.kwargs["groups"] != 1
        or not is_canonical_block(ir, option.load_block_nid)
        or ir.tree.block(option.load_block_nid).alloc_buffers
        or ir.tree.block(option.load_block_nid).annotations
        or intersects_software_pipeline(ir, (previous, option.load_block_nid))
        or configured_program_shards(ir).keys() & set(ir.tree.descendants(option.load_block_nid))
    ):
        return None
    rows, parts = store.kwargs["rows"], store.kwargs["stages"]
    if not isinstance(rows, int) or not isinstance(parts, int) or rows < 1 or parts < 2:
        return None
    source = ir.buffer(store.operand_bindings["src"].tensor)
    stored = ir.buffer(store.operand_bindings["dst"].tensor)
    output = ir.buffer(load.operand_bindings["dst"].tensor)
    if (
        len(source.shape) != 2
        or source.shape[0] != rows * parts
        or rows * parts > 128
        or source.shape[1] < 128
        or source.shape[1] % 128
        or stored.shape != (rows, parts * source.shape[1])
        or stored.location != "shared_hbm"
        or output.shape != stored.shape[::-1]
        or output.partition_extent() != 128
        or source.partition_extent() != source.shape[0]
        or source.physical_dtype() not in {"float32", "bfloat16", "float16"}
        or (source.shape[1] > 128 and source.shape[0] % (8 if source.physical_dtype() == "float32" else 16))
        or any(
            buffer.dtype != source.dtype or buffer.physical_dtype() != source.physical_dtype()
            for buffer in (stored, output)
        )
        or any(buffer.location != "sbuf" or buffer.versions != 1 or buffer.list_len != 1 for buffer in (source, output))
        or source.name == output.name
        or store.operand_bindings != {"src": _full_region(source), "dst": _full_region(stored)}
        or load.operand_bindings["src"].tensor != stored.name
        or not _store_patterns(store, source)
    ):
        return None
    writers = [
        nid for nid in ir.dependency.touches_by_tensor[stored.name] if stored.name in ir.dependency.info(nid).writes
    ]
    return (
        _Reshape(option.load_block_nid, load_leaf, source, output, rows, parts, source.shape[1])
        if writers == [store_leaf]
        else None
    )


def _rewrite_transpose(ir: KernelIR, match: _Reshape) -> None:
    """Replace one stored reshape reader with a tiled transpose and exact unpack."""
    name = f"{match.output.name}_transposed"
    while name in ir.all_buffers():
        name += "_"
    temporary = Buffer(
        name=name,
        shape=(128, match.rows * match.parts * (match.width // 128)),
        dtype=match.source.dtype,
        location="sbuf",
        storage_dtype=match.source.physical_dtype(),
        partition_size=128,
    )
    append_root_buffers(ir, (temporary,))
    extents = axis_extents(ir)
    mapping = {}
    serial = 0
    for role, size in (("K", 128), ("T", match.width // 128), ("R", match.rows), ("S", match.parts)):
        while f"d{serial}" in extents:
            serial += 1
        mapping[role] = f"d{serial}"
        extents[mapping[role]] = size
    partition_axis = f"d{serial + 1}"
    while partition_axis in extents:
        serial += 1
        partition_axis = f"d{serial + 1}"
    extents[partition_axis] = match.rows * match.parts
    tiles = match.width // 128
    temporary_pitch = temporary.per_tile_physical_shape()[2]
    output_width = match.output.per_tile_physical_shape()[2]
    output_pitch = match.output.physical_shape()[1] * output_width

    def pattern(values: tuple[tuple[int, int], ...]) -> AccessPattern:
        """Build one complete constant native access pattern."""
        return AccessPattern(
            pattern=tuple((Const(value=stride), Const(value=size)) for stride, size in values), offset=Const(value=0)
        )

    first = _append_stage_transposes(
        ir,
        match.source,
        temporary,
        (match.rows * match.parts, 128, tiles),
        {"P": partition_axis, "T": mapping["T"], "F": mapping["K"]},
        extents,
    )
    second = _append(
        ir,
        ISANode(
            op_cls=NKIPartitionUnpackCopy,
            operand_bindings={"src": _full_region(temporary), "dst": _full_region(match.output)},
            kwargs={"tiles": tiles, "rows": match.rows, "stages": match.parts, "engine": "vector"},
            access_patterns={
                "src": pattern(
                    (
                        (temporary_pitch, 128),
                        (1, match.parts),
                        (match.rows * match.parts, tiles),
                        (match.parts, match.rows),
                    )
                ),
                "dst": pattern(
                    ((output_pitch, 128), (tiles * output_width, match.parts), (output_width, tiles), (1, match.rows))
                ),
            },
        ),
        mapping,
        extents,
    )
    old_nodes = tuple(ir.tree.preorder(match.load_block))
    _replace_in_parent_children(ir.tree, ir.tree.root, [match.load_block], [first, second])
    ir.tree.graph.remove_nodes_from(old_nodes)


def _rewrite(ir: KernelIR, match: _Reshape) -> None:
    """Emit the fixed two-transpose permutation without selecting a schedule."""
    names = set(ir.all_buffers())
    temporaries = []
    for stem in (f"{match.output.name}_transposed", f"{match.output.name}_reordered"):
        name = stem
        while name in names:
            name += "_"
        names.add(name)
        temporaries.append(
            Buffer(
                name=name,
                shape=(match.width, match.rows * match.parts),
                dtype="float32",
                location="sbuf",
                storage_dtype="float32",
                partition_size=match.width,
            )
        )
    first, second = temporaries
    append_root_buffers(ir, (first, second))
    extents = axis_extents(ir)
    serial = 0

    def axes(sizes: tuple[int, ...]) -> tuple[str, ...]:
        """Allocate globally distinct concrete axes for complete native views."""
        nonlocal serial
        result = []
        for size in sizes:
            while f"d{serial}" in extents:
                serial += 1
            name = f"d{serial}"
            extents[name] = size
            result.append(name)
        return tuple(result)

    p, f = axes((match.rows * match.parts, match.width))
    first_block = _append(
        ir,
        ISANode(
            op_cls=NKIDMATranspose, operand_bindings={"src": _full_region(match.source), "dst": _full_region(first)}
        ),
        {"P": p, "F": f},
        extents,
    )
    p, f, n = axes((match.width, match.rows * match.parts, match.rows * match.parts))
    second_block = _append(
        ir,
        ISANode(
            op_cls=NKIStridedTensorCopy,
            operand_bindings={"src": _full_region(first), "dst": _full_region(second)},
            kwargs={
                "pattern": ((1, match.parts), (match.parts, match.rows)),
                "offset": 0,
                "width": match.rows * match.parts,
                "engine": "vector",
            },
        ),
        {"P": p, "F": f, "N": n},
        extents,
    )
    p, t, f = axes((match.width, match.parts, match.rows))
    final_block = _append_stage_transposes(
        ir, second, match.output, (match.width, match.rows, match.parts), {"P": p, "T": t, "F": f}, extents
    )
    _replace_in_parent_children(ir.tree, ir.tree.root, [match.load_block], [first_block, second_block, final_block])
    ir.tree.graph.remove_nodes_from((match.load_block, match.load_leaf))


def _append_stage_transposes(
    ir: KernelIR,
    source: Buffer,
    output: Buffer,
    shape: tuple[int, int, int],
    mapping: dict[str, str],
    extents: dict[str, int],
) -> int:
    """Emit ordinary stage transposes, leaving batching as a separate action."""
    rows, columns, stages = shape
    iteration = Var(name=f"i_{mapping['T']}_0") if stages > 1 else Const(value=0)
    block_nid = _append(
        ir,
        ISANode(
            op_cls=NKIDMATranspose,
            operand_bindings={
                "src": BufferRegion(
                    tensor=source.name,
                    ranges=(
                        (Const(value=0), Const(value=rows)),
                        (Mul(left=iteration, right=Const(value=columns)), Const(value=columns)),
                    ),
                ),
                "dst": BufferRegion(
                    tensor=output.name,
                    ranges=(
                        (Const(value=0), Const(value=columns)),
                        (Mul(left=iteration, right=Const(value=rows)), Const(value=rows)),
                    ),
                ),
            },
        ),
        mapping,
        extents,
    )
    if isinstance(iteration, Var):
        leaf = ir.tree.children(block_nid)[0]
        loop = ir.tree.add_node(ForNode(loop_var=iteration.name, extent=stages), parent=None)
        _replace_in_parent_children(ir.tree, block_nid, [leaf], [loop])
        ir.tree.graph.add_edge(loop, leaf)
        block = ir.tree.block(block_nid)
        ir.tree.graph.nodes[block_nid]["data"] = replace(
            block,
            iter_values=tuple(
                iteration if variable.axis == mapping["T"] else value
                for variable, value in zip(block.iter_vars, block.iter_values)
            ),
        )
    return block_nid


def _append(ir: KernelIR, leaf: ISANode, mapping: dict[str, str], extents: dict[str, int]) -> int:
    """Append one detached loop-free operation block with complete dependencies."""
    reads = tuple(region for slot, region in leaf.operand_bindings.items() if slot in leaf.op_cls.INPUT_OPERANDS)
    writes = tuple(region for slot, region in leaf.operand_bindings.items() if slot not in leaf.op_cls.INPUT_OPERANDS)
    block = ir.tree.add_node(
        BlockNode(
            iter_vars=tuple(
                IterVar(axis=axis, dom=(0, extents[axis]), role=AxisRole.PARALLEL) for axis in mapping.values()
            ),
            iter_values=tuple(Const(value=0) for _ in mapping),
            reads=reads,
            writes=writes,
            axis_map=mapping,
        )
    )
    ir.tree.add_node(leaf, parent=block)
    return block


__all__ = ["OnChipReshape", "OnChipReshapeOption"]
