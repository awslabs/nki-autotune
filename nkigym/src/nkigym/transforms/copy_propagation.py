"""Propagate an on-chip copy source into a co-located consumer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from weakref import WeakKeyDictionary

from nkigym.ir import Dependency, KernelIR
from nkigym.ir.arith import LE, Add, Analyzer, Const, Sub
from nkigym.ir.arith.expr import Expr, Var, substitute
from nkigym.ir.interval import regions_disjoint
from nkigym.ir.program_sharding import configured_program_shards, owning_block
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode
from nkigym.ops.base import CopyContract, SliceContract
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
    software_pipeline_overlap_nodes,
)
from nkigym.transforms.helper.canonical_rewrite import finalize_rewrite

_LOOP_ACCESS_INDEPENDENCE: WeakKeyDictionary[Dependency, dict[tuple[int, int], bool]] = WeakKeyDictionary()


@dataclass(frozen=True)
class CopyPropagationOption(TransformOption):
    """Identify one ordered copy and consumer input."""

    copy_block_nid: int
    consumer_block_nid: int
    consumer_operand: str


@dataclass(frozen=True)
class _CopyPropagationMatch:
    """Resolved copy propagation with exact source and destination regions."""

    option: CopyPropagationOption
    copy_leaf_nid: int
    consumer_leaf_nid: int
    source: BufferRegion
    copied: BufferRegion


class CopyPropagation(Transform[CopyPropagationOption]):
    """Substitute one value-preserving copy source into its consumer."""

    def analyze(self, ir: KernelIR) -> list[CopyPropagationOption]:
        """Return ordered copy-consumer pairs accepted by storage contracts."""
        options: list[CopyPropagationOption] = []
        buffers = ir.all_buffers()
        overlap_nodes = software_pipeline_overlap_nodes(ir)
        positions = {nid: index for index, nid in enumerate(ir.tree.preorder())}
        analyzer = _projection_analyzer(ir)
        for copy_leaf_nid in ir.dependency.graph.nodes:
            copy_leaf = ir.tree.isa(copy_leaf_nid)
            if not isinstance(copy_leaf.op_cls.algebraic_contract(copy_leaf.kwargs), (CopyContract, SliceContract)):
                continue
            copy_nid = owning_block(ir, copy_leaf_nid)
            for consumer_leaf_nid in ir.dependency.direct_consumers(copy_leaf_nid):
                consumer_nid = owning_block(ir, consumer_leaf_nid)
                if consumer_nid == copy_nid:
                    continue
                consumer = ir.tree.isa(consumer_leaf_nid)
                for operand in consumer.op_cls.INPUT_OPERANDS:
                    option = CopyPropagationOption(copy_nid, consumer_nid, operand)
                    if self._resolve(ir, option, buffers, overlap_nodes, positions=positions, analyzer=analyzer):
                        options.append(option)
        return options

    def apply(self, ir: KernelIR, option: CopyPropagationOption) -> KernelIR:
        """Recheck, copy, and propagate the source region into one consumer."""
        match = self._resolve(ir, option, ir.all_buffers())
        if match is None:
            raise TransformLegalityError(f"illegal CopyPropagation option: {option}")
        new_ir = copy_for_rewrite(ir)
        copied_match = self._resolve(new_ir, option, new_ir.all_buffers())
        if copied_match is None:
            raise AssertionError(f"CopyPropagation option disappeared after deepcopy: {option}")
        self._rewrite(new_ir, copied_match)
        return new_ir

    def _resolve(
        self,
        ir: KernelIR,
        option: CopyPropagationOption,
        buffers: dict[str, Buffer],
        overlap_nodes: frozenset[int] | None = None,
        ordered: bool | None = None,
        positions: dict[int, int] | None = None,
        analyzer: Analyzer | None = None,
    ) -> _CopyPropagationMatch | None:
        """Resolve an option when copy semantics, storage, and use-def all agree."""
        result: _CopyPropagationMatch | None = None
        copy_nid = option.copy_block_nid
        consumer_nid = option.consumer_block_nid
        if copy_nid not in ir.tree.graph or consumer_nid not in ir.tree.graph:
            return result
        if not isinstance(ir.tree.data(copy_nid), BlockNode) or not isinstance(ir.tree.data(consumer_nid), BlockNode):
            return result
        if intersects_software_pipeline(ir, (copy_nid, consumer_nid), overlap_nodes):
            return result
        copy_scope = tuple(nid for nid in ir.tree.ancestors(copy_nid) if isinstance(ir.tree.data(nid), ForNode))
        consumer_scope = tuple(nid for nid in ir.tree.ancestors(consumer_nid) if isinstance(ir.tree.data(nid), ForNode))
        if copy_scope != consumer_scope:
            return result
        if positions is None:
            positions = {nid: index for index, nid in enumerate(ir.tree.preorder())}
        copy_leaf_nid = _owned_leaf(ir, copy_nid)
        consumer_leaf_nid = _owned_leaf(ir, consumer_nid)
        if copy_leaf_nid is None or consumer_leaf_nid is None:
            return result
        if ordered is None:
            parent = ir.tree.parent(copy_nid)
            siblings = ir.tree.children(parent) if parent is not None else []
            sibling_ordered = (
                parent is not None
                and ir.tree.parent(consumer_nid) == parent
                and copy_nid in siblings
                and consumer_nid in siblings
                and siblings.index(copy_nid) < siblings.index(consumer_nid)
            )
            ordered = sibling_ordered or (
                copy_leaf_nid in ir.dependency.direct_producers(consumer_leaf_nid)
                and positions[copy_leaf_nid] < positions[consumer_leaf_nid]
            )
        if not ordered:
            return result
        copy_leaf = ir.tree.isa(copy_leaf_nid)
        consumer_leaf = ir.tree.isa(consumer_leaf_nid)
        if copy_leaf.access_patterns or consumer_leaf.access_patterns:
            return result
        contract = copy_leaf.op_cls.algebraic_contract(copy_leaf.kwargs)
        if not isinstance(contract, (CopyContract, SliceContract)):
            return result
        source = copy_leaf.operand_bindings.get(contract.input_operand)
        copied = copy_leaf.operand_bindings.get(contract.output_operand)
        consumed = consumer_leaf.operand_bindings.get(option.consumer_operand)
        if source is None or copied is None or consumed is None or consumed.tensor != copied.tensor:
            return result
        if isinstance(contract, SliceContract):
            if contract.start < 0 or contract.width < 1 or not 0 <= contract.axis < len(source.ranges):
                return result
            extent = source.ranges[contract.axis][1]
            if not isinstance(extent, Const) or contract.start + contract.width > extent.value:
                return result
            source = source.with_partition_aligned_slice(contract.axis, contract.start, contract.width)
        source = _project_source_region(
            _projection_analyzer(ir) if analyzer is None else analyzer, source, copied, consumed
        )
        if source is None:
            return result
        if copied.tensor in ir.param_buffers or copied.tensor in ir.return_names:
            return result
        copied_buffer = buffers[copied.tensor]
        source_buffer = buffers[source.tensor]
        accepted_locations = consumer_leaf.op_cls.INPUT_LOCATIONS.get(option.consumer_operand)
        locations = {
            operand: buffers[region.tensor].location
            for operand, region in consumer_leaf.operand_bindings.items()
            if operand in consumer_leaf.op_cls.INPUT_OPERANDS
        }
        locations[option.consumer_operand] = source_buffer.location
        dtypes = {
            operand: buffers[region.tensor].physical_dtype()
            for operand, region in consumer_leaf.operand_bindings.items()
            if operand in consumer_leaf.op_cls.INPUT_OPERANDS
        }
        dtypes[option.consumer_operand] = source_buffer.physical_dtype()
        required_dtype = consumer_leaf.op_cls.REQUIRED_INPUT_STORAGE_DTYPES.get(option.consumer_operand)
        accepted_dtypes = consumer_leaf.op_cls.INPUT_STORAGE_DTYPES.get(option.consumer_operand, frozenset())
        storage_compatible = (
            source_buffer.physical_dtype() == copied_buffer.physical_dtype()
            or source_buffer.physical_dtype() in accepted_dtypes
        )
        if (
            copied_buffer.location == "shared_hbm"
            or source_buffer.location not in {"shared_hbm", "sbuf", "psum"}
            or accepted_locations is None
            or source_buffer.location not in accepted_locations
            or source_buffer.dtype != copied_buffer.dtype
            or not storage_compatible
            or not consumer_leaf.op_cls.accepts_input_locations(locations)
            or not consumer_leaf.op_cls.accepts_input_storage_dtypes(dtypes)
            or (required_dtype is not None and source_buffer.physical_dtype() != required_dtype)
            or source.tensor == copied.tensor
            or len(source.ranges) != len(copied.ranges)
            or not _location_tile_compatible(consumer_leaf, option.consumer_operand, source, source_buffer)
        ):
            return result
        copy_block = ir.tree.block(copy_nid)
        if any(buffer.name != copied.tensor for buffer in copy_block.alloc_buffers):
            return result
        if not self._has_unique_definition(ir, copied.tensor, copy_leaf_nid, consumer_leaf_nid):
            return result
        if not source_remains_stable(
            ir, copy_leaf.operand_bindings[contract.input_operand], copy_leaf_nid, consumer_leaf_nid, positions
        ):
            return result
        result = _CopyPropagationMatch(
            option=option,
            copy_leaf_nid=copy_leaf_nid,
            consumer_leaf_nid=consumer_leaf_nid,
            source=source,
            copied=consumed,
        )
        return result

    def _has_unique_definition(self, ir: KernelIR, tensor: str, copy_leaf_nid: int, consumer_leaf_nid: int) -> bool:
        """Require one defining copy and a selected reader; other readers keep the copy live."""
        readers: list[int] = []
        writers: list[int] = []
        for nid in ir.dependency.touches_by_tensor.get(tensor, ()):
            node = ir.tree.data(nid)
            assert isinstance(node, ISANode)
            rmw_operands = node.op_cls.rmw_operands(node.kwargs)
            for slot, region in node.operand_bindings.items():
                if region.tensor != tensor:
                    continue
                if slot in node.op_cls.INPUT_OPERANDS or slot in rmw_operands:
                    readers.append(nid)
                if slot not in node.op_cls.INPUT_OPERANDS:
                    writers.append(nid)
        return consumer_leaf_nid in readers and writers == [copy_leaf_nid]

    def _rewrite(self, ir: KernelIR, match: _CopyPropagationMatch) -> None:
        """Retarget the consumer and rebuild derived metadata."""
        consumer = ir.tree.isa(match.consumer_leaf_nid)
        bindings = dict(consumer.operand_bindings)
        bindings[match.option.consumer_operand] = match.source
        ir.tree.graph.nodes[match.consumer_leaf_nid]["data"] = replace(consumer, operand_bindings=bindings)

        consumer_block = ir.tree.block(match.option.consumer_block_nid)
        replaced = False
        reads: list[BufferRegion] = []
        for region in consumer_block.reads:
            if region == match.copied and not replaced:
                reads.append(match.source)
                replaced = True
            else:
                reads.append(region)
        if not replaced:
            raise AssertionError(
                f"consumer block {match.option.consumer_block_nid} does not read {match.copied.tensor!r}"
            )
        ir.tree.graph.nodes[match.option.consumer_block_nid]["data"] = replace(consumer_block, reads=tuple(reads))
        finalize_rewrite(ir)


def source_remains_stable(
    ir: KernelIR, source: BufferRegion, copy_leaf_nid: int, consumer_leaf_nid: int, positions: dict[int, int]
) -> bool:
    """Require a live source allocation and no writes before the delayed read."""
    copy_position = positions[copy_leaf_nid]
    consumer_position = positions[consumer_leaf_nid]
    if copy_position >= consumer_position:
        return False
    consumer_ancestors = set(ir.tree.ancestors(consumer_leaf_nid))
    copy_ancestors = ir.tree.ancestors(copy_leaf_nid)
    for nid in copy_ancestors:
        node = ir.tree.data(nid)
        if nid not in consumer_ancestors and isinstance(node, BlockNode):
            if any(buffer.name == source.tensor and buffer.location != "shared_hbm" for buffer in node.alloc_buffers):
                return False
    crossed = [nid for nid in set(copy_ancestors) ^ consumer_ancestors if isinstance(ir.tree.data(nid), ForNode)]
    shards = configured_program_shards(ir)
    for nid in ir.dependency.touches_by_tensor.get(source.tensor, ()):
        position = positions[nid]
        writes = tuple(region for region in ir.dependency.info(nid).write_regions if region.tensor == source.tensor)
        if writes and copy_position < position <= consumer_position:
            return False
        ancestors = set(ir.tree.ancestors(nid))
        for loop in crossed:
            iterations = ir.tree.loop(loop).extent // shards.get(loop, 1)
            if (
                loop in ancestors
                and iterations > 1
                and any(
                    _future_write_overlaps(ir, loop, (copy_leaf_nid, source), (nid, write), iterations)
                    for write in writes
                )
            ):
                return False
    return True


def independent_loop_accesses(ir: KernelIR, loop_nid: int, leaf_nid: int) -> bool:
    """Prove that separating one leaf preserves cross-iteration memory dependencies."""
    cache = _LOOP_ACCESS_INDEPENDENCE.setdefault(ir.dependency, {})
    key = (loop_nid, leaf_nid)
    if key in cache:
        return cache[key]
    info = ir.dependency.info(leaf_nid)
    iterations = ir.tree.loop(loop_nid).extent
    for regions, writes in ((info.read_regions, False), (info.write_regions, True)):
        for region in regions:
            for other in ir.dependency.touches_by_tensor.get(region.tensor, ()):
                if other == leaf_nid or loop_nid not in ir.tree.ancestors(other):
                    continue
                other_info = ir.dependency.info(other)
                conflicts = (
                    (*other_info.read_regions, *other_info.write_regions) if writes else other_info.write_regions
                )
                for conflict in conflicts:
                    if conflict.tensor != region.tensor:
                        continue
                    endpoints = ((leaf_nid, region), (other, conflict))
                    if _future_write_overlaps(ir, loop_nid, *endpoints, iterations) or _future_write_overlaps(
                        ir, loop_nid, endpoints[1], endpoints[0], iterations
                    ):
                        cache[key] = False
                        return False
    cache[key] = True
    return True


def _owned_leaf(ir: KernelIR, block_nid: int) -> int | None:
    """Return the sole ISA leaf directly owned by one block."""
    leaves = [
        nid
        for nid in ir.tree.preorder(block_nid)
        if isinstance(ir.tree.data(nid), ISANode) and owning_block(ir, nid) == block_nid
    ]
    return leaves[0] if len(leaves) == 1 else None


def _iteration_region(
    ir: KernelIR, endpoint: tuple[int, BufferRegion], shared_loop: int, future: bool
) -> tuple[BufferRegion, dict[str, int]]:
    """Give endpoint-local loop variables distinct names below a shared iteration."""
    leaf_nid, region = endpoint
    substitutions: dict[str, Expr] = {}
    extents: dict[str, int] = {}
    nested = False
    for nid in ir.tree.ancestors(leaf_nid):
        node = ir.tree.data(nid)
        if not isinstance(node, ForNode):
            continue
        prefix = ("write" if future else "read") if nested else "shared"
        name = f"_copy_{prefix}_{nid}"
        value: Expr = Var(name=name)
        extents[name] = node.extent
        if nid == shared_loop:
            nested = True
            if future:
                value = Add(left=value, right=Add(left=Const(value=1), right=Var(name="_copy_distance")))
        substitutions[node.loop_var] = value
    ranges = tuple(
        (substitute(lower, substitutions), substitute(width, substitutions)) for lower, width in region.ranges
    )
    return replace(region, ranges=ranges), extents


def _future_write_overlaps(
    ir: KernelIR, loop_nid: int, read: tuple[int, BufferRegion], write: tuple[int, BufferRegion], iterations: int
) -> bool:
    """Prove whether any later local iteration may overwrite a saved copy source."""
    read_region, read_extents = _iteration_region(ir, read, loop_nid, False)
    write_region, write_extents = _iteration_region(ir, write, loop_nid, True)
    extents = {**read_extents, **write_extents, "_copy_distance": iterations - 1}
    buffer = ir.param_buffers.get(read_region.tensor) or ir.dependency.info(read[0]).buffers[read_region.tensor]
    return not regions_disjoint(read_region, write_region, buffer, buffer, extents)


def _projection_analyzer(ir: KernelIR) -> Analyzer:
    """Bind the projection proof's unchanged loop bounds once per analysis."""
    analyzer = Analyzer()
    extents: dict[str, int] = {}
    for nid in ir.tree.preorder():
        node = ir.tree.data(nid)
        if isinstance(node, ForNode):
            extents[node.loop_var] = max(extents.get(node.loop_var, 0), node.extent)
    for loop_var, extent in extents.items():
        analyzer.bind(loop_var, 0, extent)
    return analyzer


def _project_source_region(
    analyzer: Analyzer, source: BufferRegion, copied: BufferRegion, consumed: BufferRegion
) -> BufferRegion | None:
    """Map one contained copied-buffer subregion back to its value source."""
    if len(source.ranges) != len(copied.ranges) or len(copied.ranges) != len(consumed.ranges):
        return None
    ranges = []
    for (source_lower, source_width), (copied_lower, copied_width), (used_lower, used_width) in zip(
        source.ranges, copied.ranges, consumed.ranges
    ):
        if not analyzer.can_prove_equal(source_width, copied_width):
            return None
        offset = (
            Const(value=0)
            if analyzer.can_prove_equal(used_lower, copied_lower)
            else analyzer.simplify(Sub(left=used_lower, right=copied_lower))
        )
        end = analyzer.simplify(Add(left=offset, right=used_width))
        if not analyzer.can_prove(LE(left=Const(value=0), right=offset)) or not analyzer.can_prove(
            LE(left=end, right=copied_width)
        ):
            return None
        lower = analyzer.simplify(Add(left=source_lower, right=offset))
        ranges.append((lower, used_width))
    return BufferRegion(tensor=source.tensor, ranges=tuple(ranges))


def _location_tile_compatible(consumer: ISANode, operand: str, source: BufferRegion, source_buffer: Buffer) -> bool:
    """Enforce any operation-specific direct-HBM tile limits."""
    if source_buffer.location != "shared_hbm":
        return True
    limits = getattr(consumer.op_cls, "HBM_SOURCE_MAX_TILE_SIZE", {})
    for abstract_axis, maximum in limits.items():
        dimension = consumer.op_cls.operand_dimension(operand, abstract_axis)
        width = source.ranges[dimension][1]
        if not isinstance(width, Const) or width.value > maximum:
            return False
    return True


__all__ = ["CopyPropagation", "CopyPropagationOption"]
