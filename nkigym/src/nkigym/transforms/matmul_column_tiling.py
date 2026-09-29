"""Place independent output tiles in separate Tensor Engine column groups."""

from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, FloorDiv, Mod, Mul, affine_terms
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import AccessPattern, Buffer, BufferRegion, ISANode
from nkigym.ops.interleaved_matmul import NKIInterleavedMatmul
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_tensor import NKITensorTensor
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite
from nkigym.transforms.helper.canonical_rewrite import owning_block, replace_buffer


@dataclass(frozen=True)
class MatmulColumnTilingOption(TransformOption):
    """Choose a Tensor Engine column width for one matrix product."""

    isa_nid: int
    column_size: int


@dataclass(frozen=True)
class _Layout:
    """One output buffer and all of its tile-aligned physical accesses."""

    buffer: Buffer
    width: int
    touches: tuple[tuple[int, str], ...]


class MatmulColumnTiling(Transform[MatmulColumnTilingOption]):
    """Pack independent output columns without changing their contraction order.

    Only PSUM storage and its views change. Loop splitting, scheduling, and
    initialization remain separate actions. The compiler chooses allocation
    addresses within the selected column groups.
    """

    def analyze(self, ir: KernelIR) -> list[MatmulColumnTilingOption]:
        """Offer column groups when every PSUM access fits one existing N tile."""
        leaves = _instruction_leaves(ir)
        return [
            option
            for nid in leaves
            if ir.tree.isa(nid).op_cls in {NKIMatmul, NKIInterleavedMatmul}
            for size in (32, 64)
            if _match(ir, option := MatmulColumnTilingOption(nid, size), leaves) is not None
        ]

    def apply(self, ir: KernelIR, option: MatmulColumnTilingOption) -> KernelIR:
        """Recheck and change one product's PSUM layout and native tile position."""
        match = _match(ir, option, _instruction_leaves(ir))
        if match is None:
            raise TransformLegalityError(f"illegal MatmulColumnTiling option: {option}")
        result = copy_for_rewrite(ir)
        _apply(result, option, match)
        result.dependency = Dependency(result.tree)
        return result


def _match(ir: KernelIR, option: MatmulColumnTilingOption, leaves: list[int]) -> _Layout | None:
    """Require one narrow ordinary product and representable complete PSUM views."""
    if option.column_size not in (32, 64) or option.isa_nid not in ir.tree.graph:
        return None
    leaf = ir.tree.data(option.isa_nid)
    if not isinstance(leaf, ISANode) or leaf.op_cls not in {NKIMatmul, NKIInterleavedMatmul}:
        return None
    if (leaf.access_patterns if leaf.op_cls is NKIMatmul else "dst" in leaf.access_patterns) or {
        "tile_size",
        "tile_position",
        "is_transpose",
        "perf_mode",
    } & leaf.kwargs.keys():
        return None
    output = leaf.operand_bindings["dst"]
    buffer = ir.buffer(output.tensor)
    width = output.ranges[1][1]
    folded = buffer.logical_tile_count() > 1
    if (
        buffer.location != "psum"
        or buffer.physical_dtype() != "float32"
        or len(buffer.shape) != 2
        or buffer.versions != 1
        or buffer.list_len != 1
        or not 1 <= buffer.partition_extent() <= option.column_size
        or buffer.shape[0] % buffer.partition_extent()
        or not isinstance(width, Const)
        or not 1 <= width.value <= 512
    ):
        return None
    if folded:
        if buffer.shape[1] != width.value:
            return None
    elif (
        buffer.free_alignment != 1
        or buffer.partition_extent() != buffer.shape[0]
        or buffer.shape[1] % width.value
        or buffer.shape[1] <= width.value
    ):
        return None
    touches = []
    for nid in leaves:
        node = ir.tree.isa(nid)
        for slot, region in node.operand_bindings.items():
            if region.tensor == buffer.name:
                if not _valid_touch(node, slot, region, buffer, width.value, nid == option.isa_nid):
                    return None
                touches.append((nid, slot))
    return _Layout(buffer, width.value, tuple(touches))


def _valid_touch(node: ISANode, slot: str, region: BufferRegion, buffer: Buffer, width: int, product: bool) -> bool:
    """Require exact same-row views confined to one selected column group."""
    if slot in node.access_patterns or len(region.ranges) != 2:
        return False
    name = node.op_cls.NAME
    if encoder := getattr(node.op_cls, "native_parameters", None):
        name, _parameters = encoder(dict(node.kwargs), frozenset(node.operand_bindings))
    direct_psum_read = node.op_cls is NKITensorTensor and slot == "data1"
    if not product and name not in {"memset", "tensor_copy"} and not direct_psum_read:
        return False
    if node.op_cls.NAME == "nc_matmul" and not product:
        return False
    folded = buffer.logical_tile_count() > 1
    if region.ranges[0][1] != Const(value=buffer.partition_extent()):
        return False
    if not folded and region.ranges[0][0] != Const(value=0):
        return False
    lower, span = region.ranges[1]
    if folded:
        return (
            isinstance(lower, Const)
            and isinstance(span, Const)
            and 0 <= lower.value
            and 1 <= span.value
            and lower.value + span.value <= width
        )
    remainder = Analyzer().simplify(Mod(left=lower, right=Const(value=width)))
    affine = affine_terms(lower)
    if not isinstance(remainder, Const) and all(value % width == 0 for value in affine.values()):
        remainder = Const(value=0)
    return (
        isinstance(span, Const)
        and isinstance(remainder, Const)
        and 0 <= remainder.value
        and 1 <= span.value
        and remainder.value + span.value <= width
    )


def _apply(ir: KernelIR, option: MatmulColumnTilingOption, match: _Layout) -> None:
    """Rebase views without sharing an overwrite's 512-element bank footprint."""
    groups = 128 // option.column_size
    folded = match.buffer.logical_tile_count() > 1
    tiles = match.buffer.logical_tile_count() if folded else match.buffer.shape[1] // match.width
    accumulation = ir.tree.isa(option.isa_nid).kwargs.get("accumulate", True)
    bank_stride = match.width if accumulation is True and not folded else 512
    pitch = (tiles + groups - 1) // groups * bank_stride
    buffer = replace(match.buffer, shape=(128, pitch), partition_size=128)
    physical_pitch = buffer.per_tile_physical_shape()[2]
    replace_buffer(ir, buffer)
    analyzer = Analyzer()
    owners = set()
    for nid, slot in match.touches:
        node = ir.tree.isa(nid)
        original = node.operand_bindings[slot]
        lower, span = original.ranges[1]
        tile = original.ranges[0][0] if folded else FloorDiv(left=lower, right=Const(value=match.width))
        column = analyzer.simplify(
            Mul(left=Mod(left=tile, right=Const(value=groups)), right=Const(value=option.column_size))
        )
        free = Add(
            left=Mul(left=FloorDiv(left=tile, right=Const(value=groups)), right=Const(value=bank_stride)),
            right=lower if folded else Mod(left=lower, right=Const(value=match.width)),
        )
        pattern = AccessPattern(
            pattern=((Const(value=physical_pitch), original.ranges[0][1]), (Const(value=1), span)),
            offset=analyzer.simplify(Add(left=Mul(left=column, right=Const(value=physical_pitch)), right=free)),
        )
        region = replace(original, ranges=((Const(value=0), Const(value=128)), (Const(value=0), Const(value=pitch))))
        kwargs = dict(node.kwargs)
        if nid == option.isa_nid:
            kwargs.update(tile_size=(128, option.column_size), tile_position=column)
        ir.tree.graph.nodes[nid]["data"] = replace(
            node,
            operand_bindings={**node.operand_bindings, slot: region},
            access_patterns={**node.access_patterns, slot: pattern},
            kwargs=kwargs,
        )
        owners.add(owning_block(ir.tree, nid))
    for owner in owners:
        _refresh_regions(ir, owner)


def _refresh_regions(ir: KernelIR, owner: int) -> None:
    """Retain conservative physical bounds for every explicit PSUM view."""
    node = next(ir.tree.isa(nid) for nid in _instruction_leaves(ir) if owning_block(ir.tree, nid) == owner)
    rmw = node.op_cls.rmw_operands(node.kwargs)
    reads = tuple(
        region for slot, region in node.operand_bindings.items() if slot in node.op_cls.INPUT_OPERANDS or slot in rmw
    )
    writes = tuple(region for slot, region in node.operand_bindings.items() if slot not in node.op_cls.INPUT_OPERANDS)
    ir.tree.graph.nodes[owner]["data"] = replace(ir.tree.block(owner), reads=reads, writes=writes)


def _instruction_leaves(ir: KernelIR) -> list[int]:
    """Skip empty structural leaves without changing instruction traversal order."""
    return [nid for nid in ir.tree.leaves() if isinstance(ir.tree.data(nid), ISANode)]


__all__ = ["MatmulColumnTiling", "MatmulColumnTilingOption"]
