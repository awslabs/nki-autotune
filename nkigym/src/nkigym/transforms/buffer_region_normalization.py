"""Normalize one buffer's regions into its current local allocation frame."""

from __future__ import annotations

from dataclasses import dataclass, replace
from weakref import WeakKeyDictionary

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Expr, from_affine, substitute, to_affine
from nkigym.ir.dependency import Dependency
from nkigym.ir.graph_index import ordered_tree_topology
from nkigym.ir.tree import AccessPattern, BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.search.buffer_placement import _anchor_loop_nids_from_regions, _regions_by_tensor
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite

_RegionFingerprint = tuple[str, int, int, tuple[tuple[Expr, Expr], ...]]
_Normalization = tuple[int, int]
_NORMALIZATIONS: WeakKeyDictionary[KernelTree, dict[str, frozenset[_Normalization]]] = WeakKeyDictionary()


@dataclass(frozen=True)
class BufferRegionNormalizationOption(TransformOption):
    """Remove one allocation-frame loop from one physical buffer axis."""

    tensor: str
    axis: int
    anchor_loop_nid: int


class BufferRegionNormalization(Transform[BufferRegionNormalizationOption]):
    """Translate one buffer's accesses into its current local allocation frame."""

    def analyze(self, ir: KernelIR) -> list[BufferRegionNormalizationOption]:
        """Return one option per buffer axis and allocation-frame anchor."""
        tensors = frozenset(name for name, buffer in ir.all_buffers().items() if buffer.location in {"sbuf", "psum"})
        changed = _normalizations_required(ir.tree, tensors)
        return [
            BufferRegionNormalizationOption(tensor=tensor, axis=axis, anchor_loop_nid=anchor)
            for tensor, axis, anchor in sorted(changed)
            if access_patterns_fit_buffer(
                ir.tree, tensor, ir.buffer(tensor), {ir.tree.loop(anchor).loop_var: Const(value=0)}
            )
        ]

    def apply(self, ir: KernelIR, option: BufferRegionNormalizationOption) -> KernelIR:
        """Re-check legality and normalize one buffer without changing its shape."""
        if option not in self.analyze(ir):
            raise TransformLegalityError(f"illegal BufferRegionNormalization option: {option}")
        new_ir = copy_for_rewrite(ir)
        _normalize_region_axis(new_ir.tree, option.tensor, option.axis, option.anchor_loop_nid)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir


def _regions_requiring_normalization(tree: KernelTree, tensors: frozenset[str]) -> set[str]:
    """Return selected tensors whose current regions differ from normalized regions."""
    return {tensor for tensor, _axis, _anchor in _normalizations_required(tree, tensors)}


def _normalizations_required(tree: KernelTree, tensors: frozenset[str]) -> set[tuple[str, int, int]]:
    """Return tensor-axis-anchor triples that differ from the allocation frame."""
    cached = _NORMALIZATIONS.setdefault(tree, {})
    missing = tensors - cached.keys()
    if missing:
        selected = frozenset(missing)
        records = _region_fingerprints(tree, selected)
        declarations = {
            buffer.name: (block_nid, buffer)
            for block_nid in tree.blocks()
            for buffer in tree.block(block_nid).alloc_buffers
            if buffer.name in selected
        }
        ancestors = ordered_tree_topology(tree.graph, tree.root)[1]
        regions = _regions_by_tensor(tree, selected)
        for tensor, (owner, buffer) in declarations.items():
            cached[tensor] = frozenset(
                (axis, anchor)
                for anchor in _frame_anchor_nids_from_facts(tree, owner, regions.get(tensor, []), ancestors)
                for axis in range(len(buffer.shape))
                if _axis_changes(records[tensor], axis, {tree.loop(anchor).loop_var: Const(value=0)})
            )
    return {(tensor, axis, anchor) for tensor in tensors for axis, anchor in cached[tensor]}


def _normalize_region_axis(tree: KernelTree, tensor: str, axis: int, anchor_loop_nid: int) -> None:
    """Remove one allocation selector from one tensor region axis."""
    if anchor_loop_nid not in _frame_anchor_nids(tree, tensor):
        raise ValueError(f"{tensor}: loop {anchor_loop_nid} is not an enclosing allocation selector")
    _rewrite_region_axis(tree, tensor, axis, {tree.loop(anchor_loop_nid).loop_var: Const(value=0)})


def _frame_anchor_nids(tree: KernelTree, tensor: str) -> tuple[int, ...]:
    """Return loops selecting independent instances of one buffer."""
    owners = [
        block_nid
        for block_nid in tree.blocks()
        if any(buffer.name == tensor for buffer in tree.block(block_nid).alloc_buffers)
    ]
    if len(owners) != 1:
        raise ValueError(f"{tensor}: expected one declaring block, found {owners}")
    ancestors = ordered_tree_topology(tree.graph, tree.root)[1]
    pairs = _regions_by_tensor(tree, frozenset((tensor,))).get(tensor, [])
    return _frame_anchor_nids_from_facts(tree, owners[0], pairs, ancestors)


def _frame_anchor_nids_from_facts(
    tree: KernelTree, owner: int, pairs: list[tuple[int, BufferRegion]], ancestors: dict[int, tuple[int, ...]]
) -> tuple[int, ...]:
    """Return allocation selectors using one shared topology and region index."""
    anchors = _anchor_loop_nids_from_regions(tree, pairs, ancestors)
    enclosing = set(ancestors[owner])
    direct = {
        loop_nid
        for loop_nid in anchors
        if owner in ancestors[loop_nid]
        and all(
            isinstance(tree.data(nid), ForNode) for nid in ancestors[loop_nid][ancestors[loop_nid].index(owner) + 1 :]
        )
    }
    return tuple(loop_nid for loop_nid in anchors if loop_nid in enclosing or loop_nid in direct)


def _rewrite_region_axis(tree: KernelTree, tensor: str, axis: int, substitutions: dict[str, Expr]) -> None:
    """Apply one common allocation-frame translation to every tensor access."""

    def rewrite(region: BufferRegion) -> BufferRegion:
        """Rewrite only the selected tensor axis."""
        if region.tensor != tensor or axis >= len(region.ranges):
            return region
        ranges = list(region.ranges)
        lower, width = ranges[axis]
        ranges[axis] = (substitute(lower, substitutions), width)
        return replace(region, ranges=tuple(ranges))

    def rewrite_pattern(pattern: AccessPattern) -> AccessPattern:
        """Translate one explicit physical view into the same allocation frame."""
        return replace(
            pattern,
            pattern=tuple(
                (substitute(stride, substitutions), substitute(extent, substitutions))
                for stride, extent in pattern.pattern
            ),
            offset=substitute(pattern.offset, substitutions),
        )

    for block_nid in tree.blocks():
        block = tree.block(block_nid)
        reads = tuple(rewrite(region) for region in block.reads)
        writes = tuple(rewrite(region) for region in block.writes)
        if reads != block.reads or writes != block.writes:
            tree.graph.nodes[block_nid]["data"] = replace(block, reads=reads, writes=writes)
    for isa_nid in tree.preorder():
        isa = tree.data(isa_nid)
        if not isinstance(isa, ISANode):
            continue
        bindings = {slot: rewrite(region) for slot, region in isa.operand_bindings.items()}
        patterns = {
            slot: rewrite_pattern(pattern) if bindings[slot].tensor == tensor else pattern
            for slot, pattern in isa.access_patterns.items()
        }
        if bindings != isa.operand_bindings or patterns != isa.access_patterns:
            tree.graph.nodes[isa_nid]["data"] = replace(isa, operand_bindings=bindings, access_patterns=patterns)


def _axis_fingerprint(
    records: tuple[_RegionFingerprint, ...], axis: int
) -> tuple[tuple[str, int, int, Expr, Expr], ...]:
    """Return one physical axis from stable region records."""
    return tuple(
        (side, nid, index, ranges[axis][0], ranges[axis][1])
        for side, nid, index, ranges in records
        if axis < len(ranges)
    )


def _axis_changes(records: tuple[_RegionFingerprint, ...], axis: int, substitutions: dict[str, Expr]) -> bool:
    """Return whether one common frame translation changes the selected axis."""
    return any(substitute(ranges[axis][0], substitutions) != ranges[axis][0] for _side, _nid, _index, ranges in records)


def _region_fingerprints(tree: KernelTree, tensors: frozenset[str]) -> dict[str, tuple[_RegionFingerprint, ...]]:
    """Return stable per-tensor block and operand region records."""
    records: dict[str, list[_RegionFingerprint]] = {tensor: [] for tensor in tensors}
    for block_nid in tree.blocks():
        block = tree.block(block_nid)
        for side, regions in (("read", block.reads), ("write", block.writes)):
            for index, region in enumerate(regions):
                if region.tensor in tensors:
                    records[region.tensor].append((side, block_nid, index, region.ranges))
    for isa_nid in tree.preorder():
        isa = tree.data(isa_nid)
        if not isinstance(isa, ISANode):
            continue
        for index, region in enumerate(isa.operand_bindings.values()):
            if region.tensor in tensors:
                records[region.tensor].append(("operand", isa_nid, index, region.ranges))
    return {tensor: tuple(values) for tensor, values in records.items()}


def access_patterns_fit_buffer(
    tree: KernelTree,
    tensor: str,
    buffer: Buffer,
    substitutions: dict[str, Expr] | None = None,
    prior: Buffer | None = None,
) -> bool:
    """Return whether one tensor's explicit views fit one list-of-one allocation."""
    old = buffer if prior is None else prior
    for nid in tree.preorder():
        node = tree.data(nid)
        if not isinstance(node, ISANode):
            continue
        for slot, pattern in node.access_patterns.items():
            if node.operand_bindings[slot].tensor != tensor:
                continue
            if buffer.list_len != 1:
                return False
            candidate = _translated_pattern(pattern, substitutions or {}, old, buffer)
            if candidate is None or not _pattern_fits(tree, nid, candidate, buffer):
                return False
    return True


def rebase_access_patterns(tree: KernelTree, tensor: str, old: Buffer, new: Buffer) -> None:
    """Rebase one tensor's physical partition stride after leading-axis compaction."""
    for nid in tree.preorder():
        node = tree.data(nid)
        if not isinstance(node, ISANode):
            continue
        patterns = dict(node.access_patterns)
        for slot, pattern in node.access_patterns.items():
            if node.operand_bindings[slot].tensor == tensor:
                translated = _translated_pattern(pattern, {}, old, new)
                if translated is None:
                    raise ValueError(f"{tensor}: access pattern does not expose its physical partition stride")
                patterns[slot] = translated
        if patterns != node.access_patterns:
            tree.graph.nodes[nid]["data"] = replace(node, access_patterns=patterns)


def _translated_pattern(
    pattern: AccessPattern, substitutions: dict[str, Expr], old: Buffer, new: Buffer
) -> AccessPattern | None:
    """Translate loop coordinates and physical allocation strides."""
    old_free = old.per_tile_physical_shape()[2]
    new_free = new.per_tile_physical_shape()[2]
    old_stride = old.logical_tile_count() * old_free
    new_stride = new.logical_tile_count() * new_free

    def translate(expr: Expr) -> Expr:
        """Rebase coefficients measured in free-axis allocation units."""
        substituted = substitute(expr, substitutions)
        coefficients = to_affine(substituted)
        return from_affine(
            {
                variable: (
                    new_stride
                    if coefficient == old_stride
                    else (
                        coefficient // old_free * new_free
                        if old_free != new_free and coefficient % old_free == 0
                        else coefficient
                    )
                )
                for variable, coefficient in coefficients.items()
            }
        )

    dimensions = tuple((translate(stride), substitute(extent, substitutions)) for stride, extent in pattern.pattern)
    first_stride, first_extent = dimensions[0]
    valid = first_stride == Const(value=new_stride) or first_stride == Const(value=0) and first_extent == Const(value=1)
    return AccessPattern(pattern=dimensions, offset=translate(pattern.offset)) if valid else None


def _pattern_fits(tree: KernelTree, nid: int, pattern: AccessPattern, buffer: Buffer) -> bool:
    """Return whether every affine view element lies within one physical allocation."""
    extents: dict[str, int] = {}
    for ancestor in tree.ancestors(nid):
        loop = tree.data(ancestor)
        if isinstance(loop, ForNode):
            if loop.loop_var in extents:
                return False
            extents[loop.loop_var] = loop.extent
    lo, hi = _affine_bounds(pattern.offset, extents)
    for stride, extent in pattern.pattern:
        if not isinstance(stride, Const) or not isinstance(extent, Const) or stride.value < 0 or extent.value < 1:
            return False
        hi += stride.value * (extent.value - 1)
    capacity = 1
    for extent in buffer.per_tile_physical_shape():
        capacity *= extent
    return lo >= 0 and hi < capacity


def _affine_bounds(expr: Expr, extents: dict[str, int]) -> tuple[int, int]:
    """Return inclusive bounds for one affine expression over enclosing loops."""
    coeffs = to_affine(expr)
    lo = coeffs.get(None, 0)
    hi = lo
    for var, coeff in coeffs.items():
        if var is None:
            continue
        if var not in extents:
            return (-(1 << 62), 1 << 62)
        span = coeff * (extents[var] - 1)
        lo, hi = (lo + span, hi) if span < 0 else (lo, hi + span)
    return lo, hi


__all__ = [
    "BufferRegionNormalization",
    "BufferRegionNormalizationOption",
    "access_patterns_fit_buffer",
    "rebase_access_patterns",
]
