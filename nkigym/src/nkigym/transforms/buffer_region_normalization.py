"""Normalize one buffer's regions into its current local allocation frame."""

from __future__ import annotations

from dataclasses import dataclass, replace
from weakref import WeakKeyDictionary

from nkigym.ir import KernelIR
from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Expr, Mod, Mul, NonAffineError, Var, from_affine, substitute, to_affine
from nkigym.ir.buffer_placement import (
    _anchor_loop_nids_from_regions,
    _regions_by_tensor,
    layout_satisfies_output_alignment,
)
from nkigym.ir.dependency import Dependency
from nkigym.ir.graph_index import ordered_tree_topology
from nkigym.ir.program_sharding import configured_program_shards, owning_block
from nkigym.ir.tree import AccessPattern, BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.ops.base import CopyContract
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite

_RegionFingerprint = tuple[str, int, int, tuple[tuple[Expr, Expr], ...]]
_Normalization = tuple[int, int]
_NORMALIZATIONS: WeakKeyDictionary[KernelTree, dict[str, frozenset[_Normalization]]] = WeakKeyDictionary()
_PROGRAM_FRAMES: WeakKeyDictionary[KernelTree, dict[str, _ProgramFrame | None]] = WeakKeyDictionary()
_AXIS_FOLDS: WeakKeyDictionary[KernelTree, tuple[BufferAxisFoldOption, ...]] = WeakKeyDictionary()
_PROGRAM_ID = "nl.program_id(0)"


@dataclass(frozen=True)
class _ProgramFrame:
    """One buffer axis covered by aligned logical-program loops."""

    axis: int
    loops: tuple[int, ...]
    programs: int
    leaf_loops: dict[int, int | None]
    block_loops: dict[int, int | None]


@dataclass(frozen=True)
class BufferRegionNormalizationOption(TransformOption):
    """Translate one allocation or logical-program frame on one buffer axis."""

    tensor: str
    axis: int
    anchor_loop_nid: int
    program_loop_nids: tuple[int, ...] = ()


@dataclass(frozen=True)
class BufferAxisFoldOption(TransformOption):
    """Reinterpret aligned free-axis slices as partition tiles."""

    tensor: str
    free_tile: int


_BufferNormalizationOption = BufferRegionNormalizationOption | BufferAxisFoldOption


class BufferRegionNormalization(Transform[_BufferNormalizationOption]):
    """Translate one buffer's accesses into its current local allocation frame."""

    def analyze(self, ir: KernelIR) -> list[_BufferNormalizationOption]:
        """Return one option per buffer axis and allocation-frame anchor."""
        tensors = frozenset(name for name, buffer in ir.all_buffers().items() if buffer.location in {"sbuf", "psum"})
        changed = _normalizations_required(ir.tree, tensors)
        options: list[_BufferNormalizationOption] = [
            BufferRegionNormalizationOption(tensor=tensor, axis=axis, anchor_loop_nid=anchor)
            for tensor, axis, anchor in sorted(changed)
            if access_patterns_fit_buffer(
                ir.tree, tensor, ir.buffer(tensor), {ir.tree.loop(anchor).loop_var: Const(value=0)}
            )
        ]
        options.extend(
            BufferRegionNormalizationOption(tensor, frame.axis, frame.loops[0], frame.loops)
            for tensor in sorted(tensors)
            if (frame := _program_frame(ir, tensor)) is not None
        )
        options.extend(_axis_fold_options(ir))
        return options

    def apply(self, ir: KernelIR, option: _BufferNormalizationOption) -> KernelIR:
        """Re-check legality and normalize one buffer coordinate frame."""
        if option not in self.analyze(ir):
            raise TransformLegalityError(f"illegal BufferRegionNormalization option: {option}")
        new_ir = copy_for_rewrite(ir)
        if isinstance(option, BufferAxisFoldOption):
            _fold_free_axis(new_ir, option)
        elif option.program_loop_nids:
            frame = _program_frame(new_ir, option.tensor)
            if frame is None or frame.loops != option.program_loop_nids:
                raise AssertionError(f"program frame disappeared after deepcopy: {option}")
            _normalize_program_regions(new_ir.tree, option.tensor, option.axis, frame)
            _PROGRAM_FRAMES.pop(new_ir.tree, None)
        else:
            _normalize_region_axis(new_ir.tree, option.tensor, option.axis, option.anchor_loop_nid)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir


def _axis_fold_options(ir: KernelIR) -> tuple[BufferAxisFoldOption, ...]:
    """Return cached free-axis coordinate folds for copy-produced buffers."""
    cached = _AXIS_FOLDS.get(ir.tree)
    if cached is None:
        buffers = {
            name: buffer
            for name, buffer in ir.all_buffers().items()
            if buffer.location in {"sbuf", "psum"}
            and len(buffer.shape) == 2
            and buffer.logical_tile_count() == 1
            and buffer.list_len == buffer.versions == 1
        }
        regions: dict[str, list[BufferRegion]] = {name: [] for name in buffers}
        copy_writers: set[str] = set()
        invalid_writers: set[str] = set()
        patterned: set[str] = set()
        semantic_offsets: set[str] = set()
        for nid in ir.tree.preorder():
            node = ir.tree.data(nid)
            if isinstance(node, BlockNode):
                for region in (*node.reads, *node.writes):
                    if region.tensor in regions:
                        regions[region.tensor].append(region)
            elif isinstance(node, ISANode):
                contract = node.op_cls.algebraic_contract(node.kwargs)
                for slot, region in node.operand_bindings.items():
                    if region.tensor not in regions:
                        continue
                    regions[region.tensor].append(region)
                    if slot not in node.op_cls.INPUT_OPERANDS:
                        if isinstance(contract, CopyContract) and slot == contract.output_operand:
                            copy_writers.add(region.tensor)
                        else:
                            invalid_writers.add(region.tensor)
                    if slot in node.access_patterns:
                        patterned.add(region.tensor)
                for abstract_axis, (_key, source_slot) in getattr(node.op_cls, "SPLIT_OFFSET_KWARGS", {}).items():
                    source = node.operand_bindings.get(source_slot)
                    if (
                        source is not None
                        and source.tensor in regions
                        and node.op_cls.operand_dimension(source_slot, abstract_axis) == 1
                    ):
                        semantic_offsets.add(source.tensor)
        options = []
        for name in sorted(buffers):
            buffer = buffers[name]
            free_tile = _foldable_free_tile(
                ir,
                buffer,
                tuple(regions[name]),
                name in copy_writers
                and name not in invalid_writers
                and name not in patterned
                and name not in semantic_offsets,
            )
            if free_tile is not None:
                options.append(BufferAxisFoldOption(tensor=name, free_tile=free_tile))
        cached = tuple(options)
        _AXIS_FOLDS[ir.tree] = cached
    return cached


def _foldable_free_tile(ir: KernelIR, buffer: Buffer, regions: tuple[BufferRegion, ...], copy_only: bool) -> int | None:
    """Return the uniform free slice width for one legal coordinate fold."""
    result = None
    widths = {
        region.ranges[1][1].value
        for region in regions
        if len(region.ranges) == 2 and isinstance(region.ranges[1][1], Const)
    }
    if copy_only and regions and len(widths) == 1:
        width = next(iter(widths))
        aligned = 0 < width < buffer.shape[1] and buffer.shape[1] % width == 0
        aligned = aligned and all(_fold_region_is_aligned(region, buffer, width) for region in regions)
        candidate = replace(buffer, shape=(buffer.shape[0] * buffer.shape[1] // width, width))
        if aligned and layout_satisfies_output_alignment(ir.tree, candidate):
            result = width
    return result


def _fold_region_is_aligned(region: BufferRegion, buffer: Buffer, width: int) -> bool:
    """Return whether one region is a complete aligned free-axis slice."""
    result = False
    if len(region.ranges) == 2:
        leading_width = region.ranges[0][1]
        free_lower, free_width = region.ranges[1]
        if leading_width == Const(value=buffer.partition_extent()) and free_width == Const(value=width):
            try:
                coefficients = to_affine(free_lower)
            except NonAffineError:
                coefficients = {}
            else:
                result = all(coefficient % width == 0 for coefficient in coefficients.values())
    return result


def _fold_free_axis(ir: KernelIR, option: BufferAxisFoldOption) -> None:
    """Reframe aligned free-axis slices as logical partition tiles."""
    buffer = ir.buffer(option.tensor)
    factor = buffer.shape[1] // option.free_tile
    replacement = replace(buffer, shape=(buffer.shape[0] * factor, option.free_tile))

    def rewrite(region: BufferRegion) -> BufferRegion:
        """Rewrite one selected region into the folded coordinate frame."""
        result = region
        if region.tensor == option.tensor:
            free_lower, free_width = region.ranges[1]
            quotient = from_affine(
                {variable: coefficient // option.free_tile for variable, coefficient in to_affine(free_lower).items()}
            )
            tile_lower = Add(left=Mul(left=region.ranges[0][0], right=Const(value=factor)), right=quotient)
            result = replace(region, ranges=((tile_lower, region.ranges[0][1]), (Const(value=0), free_width)))
        return result

    for nid in ir.tree.preorder():
        node = ir.tree.data(nid)
        if isinstance(node, BlockNode):
            allocations = tuple(replacement if item.name == option.tensor else item for item in node.alloc_buffers)
            ir.tree.graph.nodes[nid]["data"] = replace(
                node,
                reads=tuple(rewrite(region) for region in node.reads),
                writes=tuple(rewrite(region) for region in node.writes),
                alloc_buffers=allocations,
            )
        elif isinstance(node, ISANode):
            bindings = {slot: rewrite(region) for slot, region in node.operand_bindings.items()}
            ir.tree.graph.nodes[nid]["data"] = replace(node, operand_bindings=bindings)


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


def _program_frame(ir: KernelIR, tensor: str) -> _ProgramFrame | None:
    """Return an aligned program-local partition frame for one buffer."""
    cached = _PROGRAM_FRAMES.setdefault(ir.tree, {})
    if tensor not in cached:
        cached[tensor] = _compute_program_frame(ir, tensor)
    return cached[tensor]


def _compute_program_frame(ir: KernelIR, tensor: str) -> _ProgramFrame | None:
    """Prove that every access covers one matching per-program buffer slice."""
    buffer = ir.buffer(tensor)
    pairs = _regions_by_tensor(ir.tree, frozenset((tensor,))).get(tensor, [])
    shards = configured_program_shards(ir)
    if buffer.location not in {"sbuf", "psum"} or buffer.list_len != 1 or not pairs or not shards:
        return None
    leaf_loops: dict[int, int | None] = {}
    block_loops: dict[int, int | None] = {}
    axes: set[int] = set()
    spans: set[int] = set()
    program_counts: set[int] = set()
    shard_programs = set(shards.values())
    if len(shard_programs) != 1:
        return None
    direct_programs = next(iter(shard_programs))
    for leaf_nid, region in pairs:
        if not region.ranges:
            return None
        matches: list[tuple[int, Expr, int | None, int, int, int]] = [
            (axis, lower, loop_nid, programs, coefficients.get(ir.tree.loop(loop_nid).loop_var, 0), width.value)
            for axis, (lower, width) in enumerate(region.ranges)
            if isinstance(width, Const)
            for coefficients in (_affine_or_none(lower),)
            if coefficients is not None
            for loop_nid, programs in shards.items()
            if loop_nid in ir.tree.ancestors(leaf_nid)
            and coefficients.get(ir.tree.loop(loop_nid).loop_var, 0) > 0
            and (axis != 0 or width.value == buffer.partition_extent())
        ]
        matches.extend(
            (axis, lower, None, direct_programs, coefficients[_PROGRAM_ID], width.value)
            for axis, (lower, width) in enumerate(region.ranges)
            if isinstance(width, Const)
            for coefficients in (_affine_or_none(lower),)
            if coefficients is not None
            and coefficients.get(_PROGRAM_ID, 0) > 0
            and (axis != 0 or width.value == buffer.partition_extent())
        )
        if len(matches) != 1:
            return None
        axis, lower, loop_nid, programs, coefficient, width = matches[0]
        if loop_nid is None:
            local_extent = 1
            normalized = substitute(lower, {_PROGRAM_ID: Const(value=0)})
        else:
            loop = ir.tree.loop(loop_nid)
            local_extent = loop.extent // programs
            if loop.extent % programs:
                return None
            local = Mod(left=Var(name=loop.loop_var), right=Const(value=local_extent))
            normalized = substitute(lower, {loop.loop_var: local})
        leaf = ir.tree.isa(leaf_nid)
        patterns = tuple(
            leaf.access_patterns[slot]
            for slot, binding in leaf.operand_bindings.items()
            if binding.tensor == tensor and slot in leaf.access_patterns
        )
        if patterns and (
            loop_nid is not None
            or any(
                not _pattern_fits(
                    ir.tree, leaf_nid, _substitute_pattern(pattern, {_PROGRAM_ID: Const(value=0)}), buffer
                )
                for pattern in patterns
            )
        ):
            return None
        analyzer = Analyzer()
        for ancestor in ir.tree.ancestors(leaf_nid):
            if isinstance((node := ir.tree.data(ancestor)), ForNode):
                analyzer.bind(node.loop_var, 0, node.extent)
        lo, hi = analyzer.const_int_bound(normalized)
        span = coefficient * local_extent
        if lo is None or hi is None or lo < 0 or hi >= span or (axis != 0 and hi + width > span):
            return None
        owner = owning_block(ir, leaf_nid)
        if owner in block_loops and block_loops[owner] != loop_nid:
            return None
        leaf_loops[leaf_nid] = loop_nid
        block_loops[owner] = loop_nid
        axes.add(axis)
        spans.add(span)
        program_counts.add(programs)
    if len(axes) != 1 or len(spans) != 1 or len(program_counts) != 1:
        return None
    axis = next(iter(axes))
    span = next(iter(spans))
    programs = next(iter(program_counts))
    capacity = buffer.logical_tile_count() if axis == 0 else buffer.shape[axis]
    if span * programs != capacity:
        return None
    loops = tuple(sorted(loop_nid for loop_nid in set(leaf_loops.values()) if loop_nid is not None))
    return _ProgramFrame(axis, loops, programs, leaf_loops, block_loops) if loops else None


def _affine_or_none(expr: Expr) -> dict[str | None, int] | None:
    """Return affine coefficients, or ``None`` for a normalized expression."""
    try:
        result = to_affine(expr)
    except NonAffineError:
        result = None
    return result


def _normalize_program_regions(tree: KernelTree, tensor: str, axis: int, frame: _ProgramFrame) -> None:
    """Rewrite one private buffer axis into each program's local coordinates."""

    def rewrite(region: BufferRegion, loop_nid: int | None) -> BufferRegion:
        """Translate one region with its enclosing program loop."""
        if region.tensor != tensor or axis >= len(region.ranges):
            return region
        substitutions: dict[str, Expr]
        if loop_nid is None:
            substitutions = {_PROGRAM_ID: Const(value=0)}
        else:
            loop = tree.loop(loop_nid)
            local_extent = loop.extent // frame.programs
            substitutions = {loop.loop_var: Mod(left=Var(name=loop.loop_var), right=Const(value=local_extent))}
        ranges = list(region.ranges)
        lower, width = ranges[axis]
        ranges[axis] = (substitute(lower, substitutions), width)
        return replace(region, ranges=tuple(ranges))

    def rewrite_pattern(pattern: AccessPattern, loop_nid: int | None) -> AccessPattern:
        """Translate one explicit view with the same program selector."""
        if loop_nid is not None:
            raise AssertionError("program-local access patterns require direct program indexing")
        return _substitute_pattern(pattern, {_PROGRAM_ID: Const(value=0)})

    for block_nid, loop_nid in frame.block_loops.items():
        block = tree.block(block_nid)
        tree.graph.nodes[block_nid]["data"] = replace(
            block,
            reads=tuple(rewrite(region, loop_nid) for region in block.reads),
            writes=tuple(rewrite(region, loop_nid) for region in block.writes),
        )
    for leaf_nid, loop_nid in frame.leaf_loops.items():
        leaf = tree.isa(leaf_nid)
        bindings = {slot: rewrite(region, loop_nid) for slot, region in leaf.operand_bindings.items()}
        patterns = {
            slot: rewrite_pattern(pattern, loop_nid) if bindings[slot].tensor == tensor else pattern
            for slot, pattern in leaf.access_patterns.items()
        }
        tree.graph.nodes[leaf_nid]["data"] = replace(leaf, operand_bindings=bindings, access_patterns=patterns)


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

    try:
        dimensions = tuple((translate(stride), substitute(extent, substitutions)) for stride, extent in pattern.pattern)
        offset = translate(pattern.offset)
    except NonAffineError:
        return None
    first_stride, first_extent = dimensions[0]
    valid = first_stride == Const(value=new_stride) or first_stride == Const(value=0) and first_extent == Const(value=1)
    return AccessPattern(pattern=dimensions, offset=offset) if valid else None


def _substitute_pattern(pattern: AccessPattern, substitutions: dict[str, Expr]) -> AccessPattern:
    """Apply one coordinate substitution to an explicit physical view."""
    return replace(
        pattern,
        pattern=tuple(
            (substitute(stride, substitutions), substitute(extent, substitutions)) for stride, extent in pattern.pattern
        ),
        offset=substitute(pattern.offset, substitutions),
    )


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
    "BufferAxisFoldOption",
    "BufferRegionNormalization",
    "BufferRegionNormalizationOption",
    "access_patterns_fit_buffer",
    "rebase_access_patterns",
]
