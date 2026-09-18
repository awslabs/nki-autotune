"""Place a declaration in a safe scope or reuse an existing on-chip allocation."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from operator import attrgetter
from weakref import WeakKeyDictionary, WeakValueDictionary

from nkigym.ir import KernelIR
from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Mul, Sub, Var
from nkigym.ir.buffer_placement import buffer_placement_targets, place_buffer
from nkigym.ir.dependency import Dependency
from nkigym.ir.dependency_rebind import rebind_unchanged_dependency
from nkigym.ir.program_sharding import configured_program_shards
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.ops.base import CopyContract, PermutationContract, PointwiseContract, ReductionContract
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite
from nkigym.transforms.buffer_region_normalization import preserve_unaffected_normalizations

_StorageKey = tuple[str, tuple[int, ...], str, str, int]


@dataclass(frozen=True)
class BufferPlacementOption(TransformOption):
    """Move one declaration or place its values in an existing on-chip allocation.

    Attributes:
        tensor: Buffer name whose declaration should move.
        scope_nid: An ancestor block of the lifetime-safe LCA; None selects the LCA.
        reuse_tensor: Existing storage to reuse; cannot be combined with scope_nid.
    """

    tensor: str
    scope_nid: int | None = None
    reuse_tensor: str | None = None


_OPTIONS: WeakValueDictionary[tuple[str, int | None, str | None], BufferPlacementOption] = WeakValueDictionary()
_REUSE_OPTIONS: WeakKeyDictionary[Dependency, dict[str, tuple[BufferPlacementOption, ...]]] = WeakKeyDictionary()


def _option(tensor: str, scope_nid: int | None = None, reuse_tensor: str | None = None) -> BufferPlacementOption:
    """Reuse an immutable option while an analysis result still holds it."""
    key = (tensor, scope_nid, reuse_tensor)
    result = _OPTIONS.get(key)
    if result is None:
        result = BufferPlacementOption(tensor, scope_nid, reuse_tensor)
        _OPTIONS[key] = result
    return result


class BufferPlacement(Transform[BufferPlacementOption]):
    """Select one buffer's allocation scope or a compatible existing allocation."""

    def analyze(self, ir: KernelIR) -> list[BufferPlacementOption]:
        """Offer the LCA placement and every enclosing allocation scope."""
        tensors = tuple(name for name, buffer in ir.all_buffers().items() if buffer.location in ("sbuf", "psum"))
        current = _declaration_blocks(ir.tree, frozenset(tensors))
        targets = buffer_placement_targets(ir.tree, tensors)
        options: list[BufferPlacementOption] = []
        for tensor, target in targets.items():
            if current[tensor] != target:
                options.append(_option(tensor))
            options.extend(
                _option(tensor, scope)
                for scope in reversed(ir.tree.ancestors(target))
                if scope != current[tensor] and isinstance(ir.tree.data(scope), BlockNode)
            )
        rows = _reuse_options(ir)
        for tensor in sorted(rows):
            options.extend(rows[tensor])
        return options

    def apply(self, ir: KernelIR, option: BufferPlacementOption) -> KernelIR:
        """Re-check legality, move one declaration on a deep copy, and rebuild dependencies."""
        self._check_legality(ir, option)
        new_ir = copy_for_rewrite(ir)
        if option.reuse_tensor is not None:
            _reuse_storage(new_ir, option.tensor, option.reuse_tensor)
        elif option.scope_nid is None:
            place_buffer(new_ir.tree, option.tensor)
        else:
            buffer = new_ir.buffer(option.tensor)
            for nid in new_ir.tree.blocks():
                block = new_ir.tree.block(nid)
                allocations = tuple(item for item in block.alloc_buffers if item.name != option.tensor)
                if nid == option.scope_nid:
                    allocations = (*allocations, buffer)
                if allocations != block.alloc_buffers:
                    new_ir.tree.graph.nodes[nid]["data"] = replace(block, alloc_buffers=allocations)
        new_ir.dependency = (
            Dependency(new_ir.tree)
            if option.reuse_tensor is not None
            else rebind_unchanged_dependency(ir.dependency, new_ir.tree)
        )
        if option.reuse_tensor is not None:
            _update_reuse_options(ir, new_ir, option.tensor, option.reuse_tensor)
        changed = (
            frozenset((option.tensor,))
            if option.reuse_tensor is None
            else frozenset((option.tensor, option.reuse_tensor))
        )
        preserve_unaffected_normalizations(ir.tree, new_ir.tree, changed)
        return new_ir

    def _check_legality(self, ir: KernelIR, option: BufferPlacementOption) -> None:
        """Reject unknown, HBM, explicit-pattern, and no-op placement choices."""
        buffers = ir.all_buffers()
        if option.tensor not in buffers:
            raise TransformLegalityError(f"BufferPlacement: no buffer named {option.tensor!r}")
        if buffers[option.tensor].location == "shared_hbm":
            raise TransformLegalityError(f"BufferPlacement: {option.tensor} is shared_hbm (must remain at root)")
        if option.reuse_tensor is not None:
            if option.scope_nid is not None or not _can_reuse(ir, option.tensor, option.reuse_tensor, _reuse_facts(ir)):
                raise TransformLegalityError(
                    f"BufferPlacement cannot reuse {option.reuse_tensor!r} for {option.tensor!r}"
                )
            return
        current = _declaration_blocks(ir.tree, frozenset({option.tensor}))[option.tensor]
        target = buffer_placement_targets(ir.tree, (option.tensor,))[option.tensor]
        if option.scope_nid is not None:
            if option.scope_nid not in ir.tree.ancestors(target) or not isinstance(
                ir.tree.data(option.scope_nid), BlockNode
            ):
                raise TransformLegalityError("BufferPlacement scope must be a block enclosing the lifetime-safe LCA")
            target = option.scope_nid
        if current == target:
            raise TransformLegalityError(f"BufferPlacement: {option.tensor} is already at its target scope (no-op)")


def _reuse_options(ir: KernelIR) -> dict[str, tuple[BufferPlacementOption, ...]]:
    """Cache immutable directed-pair rows in their public lexical order."""
    rows = _REUSE_OPTIONS.get(ir.dependency)
    if rows is None:
        facts = _reuse_facts(ir)
        groups: dict[_StorageKey, list[str]] = {}
        for tensor in sorted(facts.lifetimes):
            groups.setdefault(facts.storage_keys[tensor], []).append(tensor)
        rows = {
            tensor: tuple(
                _option(tensor, reuse_tensor=other)
                for other in groups[facts.storage_keys[tensor]]
                if _can_reuse(ir, tensor, other, facts)
            )
            for tensor in sorted(facts.lifetimes)
        }
        _REUSE_OPTIONS[ir.dependency] = rows
    return rows


def _update_reuse_options(ir: KernelIR, result: KernelIR, removed: str, retained: str) -> None:
    """Recheck pairs involving aliased storage after one reuse action.

    Every other tensor retains its accesses, declaration, shape, and lifetime.
    Dependency and lifetime facts are rebuilt before updating the candidate
    rows. Other transforms receive no inherited rows.
    """
    previous = _REUSE_OPTIONS.get(ir.dependency)
    if previous is not None:
        facts = _update_reuse_facts(ir, result, removed, retained)
        rows: dict[str, tuple[BufferPlacementOption, ...]] = {}
        for tensor in sorted(facts.lifetimes):
            if tensor == retained:
                rows[tensor] = tuple(
                    _option(tensor, reuse_tensor=other)
                    for other in sorted(facts.lifetimes)
                    if _can_reuse(result, tensor, other, facts)
                )
            else:
                kept = [option for option in previous[tensor] if option.reuse_tensor not in (removed, retained)]
                if _can_reuse(result, tensor, retained, facts):
                    index = bisect_left(kept, retained, key=attrgetter("reuse_tensor"))
                    kept.insert(index, _option(tensor, reuse_tensor=retained))
                rows[tensor] = tuple(kept)
        _REUSE_OPTIONS[result.dependency] = rows


def _declaration_blocks(tree: KernelTree, tensors: frozenset[str]) -> dict[str, int]:
    """Return the unique owning block for each selected declaration."""
    declarations: dict[str, int] = {}
    for nid in tree.blocks():
        block = tree.data(nid)
        assert isinstance(block, BlockNode)
        for buffer in block.alloc_buffers:
            if buffer.name in tensors:
                if buffer.name in declarations:
                    raise AssertionError(f"buffer {buffer.name!r} is declared by multiple blocks")
                declarations[buffer.name] = nid
    missing = tensors - declarations.keys()
    if missing:
        raise KeyError(f"buffers declared by no block: {sorted(missing)}")
    return declarations


@dataclass(frozen=True)
class _Lifetime:
    """An initialized region, its live interval, and its enclosing control scopes."""

    leaves: tuple[int, ...]
    region: BufferRegion
    interval: tuple[int, int]
    ancestors: frozenset[int]
    conditions: tuple[int, ...]
    uniform_conditions: bool


@dataclass(frozen=True)
class _ReuseFacts:
    """Storage and lifetime facts shared across candidate buffer pairs."""

    owners: dict[str, int]
    lifetimes: dict[str, _Lifetime]
    storage_keys: dict[str, _StorageKey]


_REUSE_FACTS: WeakKeyDictionary[Dependency, _ReuseFacts] = WeakKeyDictionary()


def _update_reuse_facts(ir: KernelIR, result: KernelIR, removed: str, retained: str) -> _ReuseFacts:
    """Recompute only the lifetime whose accesses changed during storage reuse."""
    prior = _reuse_facts(ir)
    order, ancestors, descendants = result.dependency._topology()
    positions = {nid: 2 * position for nid, position in order.items()}
    ends = {nid: 2 * (order[nid] + len(descendants[nid])) for nid in order}
    lifetime = _lifetime(result, result.buffer(retained), positions, ancestors, ends, configured_program_shards(result))
    lifetimes = {name: value for name, value in prior.lifetimes.items() if name not in (removed, retained)}
    storage = {name: value for name, value in prior.storage_keys.items() if name not in (removed, retained)}
    if lifetime is not None:
        lifetimes[retained] = lifetime
        storage[retained] = prior.storage_keys[retained]
    facts = _ReuseFacts({name: owner for name, owner in prior.owners.items() if name != removed}, lifetimes, storage)
    _REUSE_FACTS[result.dependency] = facts
    return facts


def _reuse_facts(ir: KernelIR) -> _ReuseFacts:
    """Cache immutable facts until dependencies are rebuilt."""
    cached = _REUSE_FACTS.get(ir.dependency)
    if cached is None:
        buffers = {
            name: buffer
            for name, buffer in ir.all_buffers().items()
            if buffer.location in {"sbuf", "psum"}
            and buffer.versions == buffer.list_len == 1
            and name not in ir.param_buffers
            and name not in ir.return_names
        }
        order, ancestors, descendants = ir.dependency._topology()
        positions = {nid: 2 * position for nid, position in order.items()}
        subtree_ends = {nid: 2 * (order[nid] + len(descendants[nid])) for nid in order}
        shards = configured_program_shards(ir)
        lifetimes = {}
        for tensor, buffer in buffers.items():
            lifetime = _lifetime(ir, buffer, positions, ancestors, subtree_ends, shards)
            if lifetime is not None:
                lifetimes[tensor] = lifetime
        storage_keys = {
            name: (buffer.location, buffer.shape, buffer.dtype, buffer.physical_dtype(), buffer.partition_extent())
            for name, buffer in buffers.items()
            if name in lifetimes
        }
        cached = _ReuseFacts(_declaration_blocks(ir.tree, frozenset(buffers)), lifetimes, storage_keys)
        _REUSE_FACTS[ir.dependency] = cached
    return cached


def _lifetime(
    ir: KernelIR,
    buffer: Buffer,
    positions: dict[int, int],
    ancestors: dict[int, tuple[int, ...]],
    subtree_ends: dict[int, int],
    shards: dict[int, int],
) -> _Lifetime | None:
    """Keep covered values live through all nested iterations that read them.

    Instruction positions are even; a loop's synthetic exit is odd. This
    distinguishes completion of repeated reads from a single final instruction,
    which alone may justify an in-place overwrite at a shared endpoint.
    Resolve bounds from each leaf's enclosing loops, including loops outside
    its own block after code motion. Initialization must dominate every guard;
    guarded uses remain live over their full enclosing iterations. Mixed guards
    permit only disjoint-lifetime reuse, never a shared instruction endpoint.
    Register sources remain live through captured uses because engine sequencers
    can materialize a virtual register after its source-level load.
    """
    tensor = buffer.name
    leaves = tuple(sorted(set(ir.dependency.touches_by_tensor.get(tensor, ())), key=positions.__getitem__))
    if not leaves or tensor in ir.dependency.info(leaves[0]).reads:
        return None
    scopes = tuple(tuple(nid for nid in ancestors[leaf] if isinstance(ir.tree.data(nid), ForNode)) for leaf in leaves)
    conditions = tuple(
        tuple(
            nid
            for nid in ancestors[leaf]
            if isinstance(owner := ir.tree.data(nid), BlockNode) and "predicate" in owner.annotations
        )
        for leaf in leaves
    )
    if any(condition[: len(conditions[0])] != conditions[0] for condition in conditions[1:]):
        return None
    starts = [index for index, leaf in enumerate(leaves) if tensor not in ir.dependency.info(leaf).reads]
    if any(conditions[index] != conditions[0] for index in starts):
        return None
    beginning = positions[leaves[0]]
    end = positions[leaves[-1]]
    complete_region = None
    for start, stop in zip(starts, [*starts[1:], len(leaves)], strict=True):
        loops = scopes[start]
        written = next(region for region in ir.dependency.info(leaves[start]).write_regions if region.tensor == tensor)
        if any(scope[: len(loops)] != loops for scope in scopes[start:stop]):
            definition = _complete_free_axis_write(
                ir, buffer, leaves[start:stop], scopes[start:stop], positions, shards
            )
            if definition is None:
                definition = _complete_tiled_write(
                    ir, buffer, leaves[start:stop], scopes[start:stop], positions, shards
                )
            if definition is None:
                return None
            loops, written, phase_beginning = definition
            beginning = min(beginning, phase_beginning)
            complete_region = written
        for scope in scopes[start:stop]:
            if len(scope) > len(loops):
                end = max(end, subtree_ends[scope[len(loops)]] + 1)
        for leaf, scope in zip(leaves[start:stop], scopes[start:stop], strict=True):
            info = ir.dependency.info(leaf)
            extents = info.extents | {ir.tree.loop(nid).loop_var: ir.tree.loop(nid).extent for nid in scope}
            if any(
                not _written_tile_covers(written, region, extents, buffer)
                for region in (*info.read_regions, *info.write_regions)
                if region.tensor == tensor
            ):
                return None
    regions = []
    for leaf in leaves:
        node = ir.tree.isa(leaf)
        if node.op_cls.OUTPUT_LOCATION == "register" and tensor in ir.dependency.info(leaf).reads:
            for register in ir.dependency.info(leaf).writes:
                for consumer in ir.dependency.touches_by_tensor[register]:
                    end = max(end, positions[consumer])
                    for scope in ancestors[consumer]:
                        if scope not in ancestors[leaf] and isinstance(ir.tree.data(scope), ForNode):
                            end = max(end, subtree_ends[scope] + 1)
        if getattr(node.op_cls, "SCALAR_OFFSET_COPY", False) and node.operand_bindings["offset"].tensor == tensor:
            """Indirect offsets remain live until their enclosing execution scope completes."""
            scope = next(
                (
                    nid
                    for nid in reversed(ancestors[leaf])
                    if isinstance(ir.tree.data(nid), ForNode)
                    or isinstance(owner := ir.tree.data(nid), BlockNode)
                    and "predicate" in owner.annotations
                ),
                ir.tree.root,
            )
            end = max(end, subtree_ends[scope] + 1)
        if node.access_patterns or any(
            isinstance((owner := ir.tree.data(nid)), BlockNode)
            and owner.annotations.keys() - {"program_shards", "predicate"}
            for nid in ancestors[leaf]
        ):
            return None
        info = ir.dependency.info(leaf)
        regions.extend(region for region in (*info.read_regions, *info.write_regions) if region.tensor == tensor)
    if not regions:
        return None
    shared_ancestors = frozenset(ancestors[leaves[0]]).intersection(*(ancestors[leaf] for leaf in leaves[1:]))
    return _Lifetime(
        leaves,
        complete_region or regions[0],
        (beginning, end),
        shared_ancestors,
        conditions[0],
        all(condition == conditions[0] for condition in conditions),
    )


def _complete_free_axis_write(
    ir: KernelIR,
    buffer: Buffer,
    leaves: tuple[int, ...],
    scopes: tuple[tuple[int, ...], ...],
    positions: dict[int, int],
    shards: dict[int, int],
) -> tuple[tuple[int, ...], BufferRegion, int] | None:
    """Certify dense initialization before subsequent reads or in-place updates.

    The shared guard accounts for implicit control-register dependencies.
    Only the operation's direct data operands must use on-chip tensor memory.
    """
    if not scopes[0] or len(buffer.shape) != 2 or buffer.shape[0] != buffer.partition_extent():
        return None
    loop_nid, outer = scopes[0][-1], scopes[0][:-1]
    if loop_nid in shards or any(loop_nid in scope or scope[: len(outer)] != outer for scope in scopes[1:]):
        return None
    if any(buffer.name in ir.dependency.info(nid).writes - ir.dependency.info(nid).reads for nid in leaves[1:]):
        return None
    leaf = ir.tree.isa(leaves[0])
    contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
    if (
        not isinstance(contract, (CopyContract, PermutationContract, PointwiseContract))
        or "program_ownership" in leaf.kwargs
    ):
        return None
    output = contract.output_operand
    if leaf.op_cls.operand_axis_groups(output) not in ((("P",), ("F",)), (("F",), ("P",))):
        return None
    inputs = ir.dependency.info(leaves[0])
    operands = (region.tensor for slot, region in leaf.operand_bindings.items() if slot in leaf.op_cls.INPUT_OPERANDS)
    if any(
        tensor not in inputs.buffers or inputs.buffers[tensor].location not in {"sbuf", "psum"} for tensor in operands
    ):
        return None
    region = leaf.operand_bindings[output]
    loop = ir.tree.loop(loop_nid)
    lower, width = region.ranges[1]
    complete = (
        region.tensor == buffer.name
        and region.ranges[0] == (Const(value=0), Const(value=buffer.shape[0]))
        and isinstance(width, Const)
        and width.value > 0
        and width.value * loop.extent == buffer.shape[1]
        and Analyzer().can_prove_equal(lower, Mul(left=Var(name=loop.loop_var), right=width))
    )
    return (
        (
            outer,
            replace(region, ranges=(region.ranges[0], (Const(value=0), Const(value=buffer.shape[1])))),
            positions[loop_nid],
        )
        if complete
        else None
    )


def _complete_tiled_write(
    ir: KernelIR,
    buffer: Buffer,
    leaves: tuple[int, ...],
    scopes: tuple[tuple[int, ...], ...],
    positions: dict[int, int],
    shards: dict[int, int],
) -> tuple[tuple[int, ...], BufferRegion, int] | None:
    """Prove a rectangular initialization completes before any later use."""
    shared = 0
    while shared < len(scopes[0]) and all(
        len(scope) > shared and scope[shared] == scopes[0][shared] for scope in scopes
    ):
        shared += 1
    loops = scopes[0][shared:]
    if not loops or any(nid in shards or any(nid in scope for scope in scopes[1:]) for nid in loops):
        return None
    node = ir.tree.isa(leaves[0])
    if node.access_patterns or getattr(node.op_cls, "SCALAR_OFFSET_COPY", False):
        return None
    writes = [region for region in ir.dependency.info(leaves[0]).write_regions if region.tensor == buffer.name]
    if len(writes) != 1 or len(writes[0].ranges) != len(buffer.shape):
        return None
    used: set[int] = set()
    analyzer = Analyzer()
    for axis, ((lower, width), extent) in enumerate(zip(writes[0].ranges, buffer.shape, strict=True)):
        if not isinstance(width, Const) or width.value < 1:
            return None
        if axis == 0:
            lower = Mul(left=lower, right=Const(value=buffer.partition_extent()))
        if width.value == extent and analyzer.can_prove_equal(lower, Const(value=0)):
            continue
        matches = [
            nid
            for nid in loops
            if nid not in used
            and ir.tree.loop(nid).extent * width.value == extent
            and analyzer.can_prove_equal(lower, Mul(left=Var(name=ir.tree.loop(nid).loop_var), right=width))
        ]
        if len(matches) != 1:
            return None
        used.add(matches[0])
    if used != set(loops):
        return None
    region = BufferRegion(
        tensor=buffer.name, ranges=tuple((Const(value=0), Const(value=extent)) for extent in buffer.shape)
    )
    return scopes[0][:shared], region, positions[loops[0]]


def _written_tile_covers(
    written: BufferRegion, accessed: BufferRegion, extents: dict[str, int], buffer: Buffer
) -> bool:
    """Prove every accessed coordinate stays within the initialized tile."""
    if written.ranges == accessed.ranges:
        return True
    if len(written.ranges) != len(accessed.ranges):
        return False
    analyzer = Analyzer()
    for name, extent in extents.items():
        analyzer.bind(name, 0, extent)
    for axis, ((base, capacity), (lower, width)) in enumerate(zip(written.ranges, accessed.ranges, strict=True)):
        if axis == 0:
            scale = Const(value=buffer.partition_extent())
            base, lower = Mul(left=base, right=scale), Mul(left=lower, right=scale)
        first, _ = analyzer.const_int_bound(Sub(left=lower, right=base))
        _, last = analyzer.const_int_bound(Sub(left=Add(left=lower, right=width), right=Add(left=base, right=capacity)))
        if first is None or last is None or first < 0 or last > 0:
            return False
    return True


def _can_reuse(ir: KernelIR, tensor: str, other: str, facts: _ReuseFacts) -> bool:
    """Prove ordinary lifetime separation or one explicitly supported in-place use."""
    if tensor == other or tensor not in facts.lifetimes or other not in facts.lifetimes:
        return False
    if facts.storage_keys[tensor] != facts.storage_keys[other]:
        return False
    lifetime, existing = facts.lifetimes[tensor], facts.lifetimes[other]
    if facts.owners[other] not in lifetime.ancestors:
        return False
    start, end = lifetime.interval
    other_start, other_end = existing.interval
    if end < other_start or other_end < start:
        return True
    if lifetime.conditions != existing.conditions or not lifetime.uniform_conditions or not existing.uniform_conditions:
        return False
    if lifetime.region.ranges != existing.region.ranges:
        return False
    if start == other_end and _inplace_pair(ir.tree.isa(lifetime.leaves[0]), tensor, other) is not None:
        return True
    if end == other_start and _inplace_pair(ir.tree.isa(existing.leaves[0]), other, tensor) is not None:
        return True
    return len(lifetime.leaves) == 1 and _dead_identity_copy(ir.tree.isa(lifetime.leaves[0]), tensor, other)


def _inplace_pair(node: ISANode, output: str, source: str) -> tuple[str, str] | None:
    """Resolve an output/input pair permitted to share storage by the ISA contract."""
    for output_slot, input_slots in node.op_cls.INPLACE_OPERANDS.items():
        destination = node.operand_bindings.get(output_slot)
        if destination is None or destination.tensor != output:
            continue
        for input_slot in input_slots:
            operand = node.operand_bindings.get(input_slot)
            if (
                operand is not None
                and operand.tensor == source
                and operand.ranges == destination.ranges
                and node.op_cls.operand_axis_groups(input_slot) == node.op_cls.operand_axis_groups(output_slot)
            ):
                return output_slot, input_slot
    return None


def _dead_identity_copy(node: ISANode, output: str, source: str) -> bool:
    """Allow a write-only identity copy to use its unchanged input allocation."""
    pair = _inplace_pair(node, output, source)
    contract = node.op_cls.algebraic_contract(node.kwargs)
    result = False
    if isinstance(contract, CopyContract):
        result = pair == (contract.output_operand, contract.input_operand)
    elif isinstance(contract, ReductionContract):
        result = (
            pair == (contract.mapped_output_operand, contract.input_operand)
            and contract.map_operator == "copy"
            and contract.scale == 1.0
            and contract.bias == 0.0
            and contract.bias_operand not in node.operand_bindings
            and "scale" not in node.kwargs
            and "bias" not in node.kwargs
        )
    return result


def _reuse_storage(ir: KernelIR, tensor: str, other: str) -> None:
    """Retarget one buffer's uses and remove its redundant declaration."""

    def region(value: BufferRegion) -> BufferRegion:
        """Preserve coordinates while selecting the retained allocation."""
        return replace(value, tensor=other) if value.tensor == tensor else value

    for nid in ir.tree.preorder():
        node = ir.tree.data(nid)
        if isinstance(node, BlockNode):
            if not any(value.tensor == tensor for value in (*node.reads, *node.writes)) and not any(
                buffer.name == tensor for buffer in node.alloc_buffers
            ):
                continue
            ir.tree.graph.nodes[nid]["data"] = replace(
                node,
                reads=tuple(region(value) for value in node.reads),
                writes=tuple(region(value) for value in node.writes),
                alloc_buffers=tuple(buffer for buffer in node.alloc_buffers if buffer.name != tensor),
            )
        elif isinstance(node, ISANode) and any(value.tensor == tensor for value in node.operand_bindings.values()):
            ir.tree.graph.nodes[nid]["data"] = replace(
                node, operand_bindings={slot: region(value) for slot, value in node.operand_bindings.items()}
            )


__all__ = ["BufferPlacement", "BufferPlacementOption"]
