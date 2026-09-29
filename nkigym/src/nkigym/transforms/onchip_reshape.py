"""Replace one reshape load with an equivalent on-chip permutation."""

from dataclasses import dataclass

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const
from nkigym.ir.tree import AccessPattern, BlockNode, Buffer, BufferRegion, ISANode, IterVar
from nkigym.ops.base import AxisRole
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.grouped_store import NKIGroupedStore
from nkigym.ops.load import NKILoad
from nkigym.ops.strided_tensor_copy import NKIStridedTensorCopy
from nkigym.ops.tiled_dma_transpose import NKITiledDMATranspose
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
)
from nkigym.transforms.helper.canonical_rewrite import append_root_buffers, axis_extents, finalize_rewrite, single_leaf
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
    dead is a separate EliminateDeadProducer action. Loop order, existing
    allocation scopes, arithmetic, and buffer reuse remain unchanged.
    """

    def analyze(self, ir: KernelIR) -> list[OnChipReshapeOption]:
        """Offer complete adjacent reshape loads with compatible SBUF geometry."""
        children = ir.tree.children(ir.tree.root)
        previous = dict(zip(children[1:], children))
        return [
            option
            for nid in previous
            if _match(ir, option := OnChipReshapeOption(nid), previous) is not None
            or match_packing_fold(ir, nid, previous[nid]) is not None
        ]

    def apply(self, ir: KernelIR, option: OnChipReshapeOption) -> KernelIR:
        """Recheck and replace the selected load with one on-chip reshape."""
        match = _match(ir, option)
        children = ir.tree.children(ir.tree.root)
        previous = dict(zip(children[1:], children)).get(option.load_block_nid)
        packing = match_packing_fold(ir, option.load_block_nid, previous) if match is None else None
        if match is None and packing is None:
            raise TransformLegalityError(f"illegal OnChipReshape option: {option}")
        result = copy_for_rewrite(ir)
        if match is not None:
            _rewrite(result, match)
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
    final_block = _append(
        ir,
        ISANode(
            op_cls=NKITiledDMATranspose,
            operand_bindings={"src": _full_region(second), "dst": _full_region(match.output)},
            kwargs={"tiles": match.parts, "width": match.rows, "axes": (3, 1, 2, 0)},
            access_patterns={
                "src": AccessPattern(
                    pattern=tuple(
                        (Const(value=stride), Const(value=size))
                        for stride, size in (
                            (match.rows * match.parts, match.width),
                            (match.rows, match.parts),
                            (1, 1),
                            (1, match.rows),
                        )
                    ),
                    offset=Const(value=0),
                ),
                "dst": AccessPattern(
                    pattern=tuple(
                        (Const(value=stride), Const(value=size))
                        for stride, size in (
                            (match.output.per_tile_physical_shape()[2], match.rows),
                            (match.width, match.parts),
                            (1, 1),
                            (1, match.width),
                        )
                    ),
                    offset=Const(value=0),
                ),
            },
        ),
        {"P": p, "T": t, "F": f},
        extents,
    )
    _replace_in_parent_children(ir.tree, ir.tree.root, [match.load_block], [first_block, second_block, final_block])
    ir.tree.graph.remove_nodes_from((match.load_block, match.load_leaf))


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
