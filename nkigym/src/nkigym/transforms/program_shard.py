"""Shard one output or additive-reduction axis across logical NeuronCores."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import (
    Add,
    BlockNode,
    Buffer,
    BufferRegion,
    Const,
    Expr,
    ForNode,
    ISANode,
    KernelIR,
    Var,
    substitute,
    to_affine,
)
from nkigym.ir.interval import regions_disjoint
from nkigym.ops.base import AxisRole, BilinearReductionContract, ReductionContract
from nkigym.ops.sendrecv import NKISendRecv
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_tensor import NKITensorTensor
from nkigym.search.program_sharding import (
    PROGRAM_SHARDS_ANNOTATION,
    axis_loop_for_block,
    block_has_axis,
    configured_program_shards,
    owning_block,
)
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    software_pipeline_overlap_nodes,
)
from nkigym.transforms.helper.canonical_rewrite import finalize_rewrite, fresh_name
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children


@dataclass(frozen=True)
class ProgramShardOption(TransformOption):
    """Assign one loop to two programs, or remove one isolated assignment."""

    loop_nid: int
    axis: str
    programs: int
    reduction_tensor: str | None = None


@dataclass(frozen=True)
class _ReductionMatch:
    """Resolved additive partial and its final drain/store path."""

    axis_blocks: frozenset[int]
    accumulator_leaf: int
    drain_leaf: int
    drain_block: int
    store_leaf: int
    accumulator_region: BufferRegion
    local_region: BufferRegion


@dataclass(frozen=True)
class _ShardFacts:
    """Immutable tree facts shared across one ProgramShard analysis."""

    block_nids: tuple[int, ...]
    blocks_by_loop_axis: dict[tuple[int, str], tuple[int, ...]]
    owners: dict[int, int]
    axis_loops: dict[tuple[int, str], int | None]
    buffers: dict[str, Buffer]
    configured: dict[int, int]
    pipeline_depths: dict[int, int]
    pipeline_overlap: frozenset[int]


class ProgramShard(Transform[ProgramShardOption]):
    """Partition disjoint output work or one complete additive reduction across cores."""

    SPLIT_PREPARATION_DEPTH = 1

    def split_preparation_ready(self, ir: KernelIR) -> bool:
        """Accept split preparation only for reductions or kernel-wide output axes."""
        block_nids = tuple(ir.tree.blocks())
        return any(
            option.reduction_tensor is not None
            or sum(block_has_axis(ir, block_nid, option.axis) for block_nid in block_nids) * 2 >= len(block_nids)
            for option in self.analyze(ir)
        )

    def analyze(self, ir: KernelIR) -> list[ProgramShardOption]:
        """Offer every legal single-loop output or additive-reduction LNC-2 shard."""
        options: list[ProgramShardOption] = []
        facts = _shard_facts(ir)
        candidates = sorted(
            {
                (loop_nid, iter_var.axis)
                for block_nid in facts.block_nids
                for iter_var in ir.tree.block(block_nid).iter_vars
                if iter_var.role in {AxisRole.PARALLEL, AxisRole.ACCUMULATION}
                if (loop_nid := facts.axis_loops[block_nid, iter_var.axis]) is not None
            }
        )
        reduction_options = self._reduction_options_by_candidate(ir, frozenset(candidates), facts)
        for loop_nid, axis in candidates:
            if loop_nid in facts.configured:
                option = ProgramShardOption(loop_nid=loop_nid, axis=axis, programs=1)
                if self._is_legal(ir, option, facts):
                    options.append(option)
                continue
            output_option = ProgramShardOption(loop_nid=loop_nid, axis=axis, programs=2)
            if self._is_legal(ir, output_option, facts):
                options.append(output_option)
            options.extend(reduction_options.get((loop_nid, axis), ()))
        return options

    def apply(self, ir: KernelIR, option: ProgramShardOption) -> KernelIR:
        """Re-check legality and apply one kernel-wide SPMD shard decision."""
        match = self._check_legality(ir, option, _shard_facts(ir))
        new_ir = copy_for_rewrite(ir)
        self._set_shard_annotation(new_ir, option)
        if match is not None:
            self._insert_partial_combine(new_ir, option, match)
            finalize_rewrite(new_ir)
        return new_ir

    def _reduction_options_by_candidate(
        self, ir: KernelIR, candidates: frozenset[tuple[int, str]], facts: _ShardFacts
    ) -> dict[tuple[int, str], tuple[ProgramShardOption, ...]]:
        """Index legal additive-reduction choices in one dependency traversal."""
        tensors: dict[tuple[int, str], set[str]] = {}
        for leaf_nid in ir.dependency.graph.nodes:
            leaf = ir.tree.isa(leaf_nid)
            block_nid = facts.owners[leaf_nid]
            block = ir.tree.block(block_nid)
            contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
            if not isinstance(contract, (BilinearReductionContract, ReductionContract)):
                continue
            axis = block.axis_map.get(contract.reduction_axis)
            region = leaf.operand_bindings.get(contract.output_operand)
            loop_nid = None if axis is None else facts.axis_loops[block_nid, axis]
            if (
                axis is not None
                and loop_nid is not None
                and (loop_nid, axis) in candidates
                and contract.combinator.combiner == "add"
                and region is not None
                and facts.buffers[region.tensor].location == "psum"
            ):
                tensors.setdefault((loop_nid, axis), set()).add(region.tensor)
        options: dict[tuple[int, str], tuple[ProgramShardOption, ...]] = {}
        for candidate, candidate_tensors in tensors.items():
            loop_nid, axis = candidate
            legal = []
            for tensor in sorted(candidate_tensors):
                option = ProgramShardOption(loop_nid, axis, 2, reduction_tensor=tensor)
                if self._is_legal(ir, option, facts):
                    legal.append(option)
            options[candidate] = tuple(legal)
        return options

    def _is_legal(self, ir: KernelIR, option: ProgramShardOption, facts: _ShardFacts) -> bool:
        """Return whether one option satisfies the full SPMD contract."""
        try:
            self._check_legality(ir, option, facts)
        except TransformLegalityError:
            return False
        return True

    def _check_legality(self, ir: KernelIR, option: ProgramShardOption, facts: _ShardFacts) -> _ReductionMatch | None:
        """Reject one loop shard that duplicates output or exposes partial values."""
        if option.programs == 1:
            self._check_unshard(ir, option, facts)
            return None
        if option.programs != 2:
            raise TransformLegalityError("ProgramShard currently supports exactly two logical NeuronCores")
        if option.loop_nid in facts.configured:
            raise TransformLegalityError(f"ProgramShard loop {option.loop_nid} is already sharded")
        if option.loop_nid not in ir.tree.graph or not isinstance(ir.tree.data(option.loop_nid), ForNode):
            raise TransformLegalityError(f"ProgramShard loop {option.loop_nid} is not materialized")
        nested = {*ir.tree.ancestors(option.loop_nid), *ir.tree.descendants(option.loop_nid)}
        if facts.configured.keys() & nested:
            raise TransformLegalityError("ProgramShard cannot nest one program partition inside another")
        block_nids = facts.blocks_by_loop_axis.get((option.loop_nid, option.axis), ())
        if not block_nids:
            raise TransformLegalityError(
                f"ProgramShard loop {option.loop_nid} does not materialize axis {option.axis!r}"
            )
        reduction = option.reduction_tensor is not None
        self._check_axis_roles(ir, option, block_nids, reduction)
        self._check_pipeline_extent(ir, option, facts)
        if reduction and option.loop_nid in facts.pipeline_overlap:
            raise TransformLegalityError("reduction ProgramShard cannot overlap an active software pipeline")
        axis_blocks = frozenset(block_nids)
        match = self._reduction_match(ir, option, axis_blocks, facts) if reduction else None
        self._check_dependencies(ir, option, axis_blocks, match, facts)
        self._check_hbm_writes(ir, option, axis_blocks, match, facts)
        return match

    def _check_unshard(self, ir: KernelIR, option: ProgramShardOption, facts: _ShardFacts) -> None:
        """Require one isolated output shard with no collective side effects."""
        if facts.configured != {option.loop_nid: 2} or option.reduction_tensor is not None:
            raise TransformLegalityError("ProgramShard can remove only one isolated independent shard")
        if any(
            ir.tree.isa(leaf_nid).op_cls is NKISendRecv or "program_ownership" in ir.tree.isa(leaf_nid).kwargs
            for leaf_nid in ir.dependency.graph.nodes
        ):
            raise TransformLegalityError("ProgramShard cannot remove a shard with collective or store ownership")
        blocks = facts.blocks_by_loop_axis.get((option.loop_nid, option.axis), ())
        if not blocks:
            raise TransformLegalityError("ProgramShard removal does not match the configured loop axis")

    def _check_pipeline_extent(self, ir: KernelIR, option: ProgramShardOption, facts: _ShardFacts) -> None:
        """Require enough program-local iterations for an active pipeline."""
        depth = facts.pipeline_depths.get(option.loop_nid)
        local_extent = ir.tree.loop(option.loop_nid).extent // option.programs
        if depth is not None and local_extent <= depth:
            raise TransformLegalityError(
                f"ProgramShard loop {option.loop_nid} leaves {local_extent} local iterations "
                f"for pipeline depth {depth}"
            )

    def _check_axis_roles(
        self, ir: KernelIR, option: ProgramShardOption, block_nids: tuple[int, ...], reduction: bool
    ) -> None:
        """Validate every block role carried by the selected physical loop."""
        all_roles: set[AxisRole] = set()
        allowed = {AxisRole.PARALLEL, AxisRole.ACCUMULATION} if reduction else {AxisRole.PARALLEL}
        for block_nid in block_nids:
            roles = {item.role for item in ir.tree.block(block_nid).iter_vars if item.axis == option.axis}
            if len(roles) != 1 or not roles <= allowed:
                raise TransformLegalityError(f"ProgramShard axis {option.axis!r} has unsupported roles {roles}")
            all_roles.update(roles)
        extent = ir.tree.loop(option.loop_nid).extent
        if extent % option.programs:
            raise TransformLegalityError(
                f"ProgramShard loop {option.loop_nid} extent {extent} is not divisible by {option.programs}"
            )
        if reduction and AxisRole.ACCUMULATION not in all_roles:
            raise TransformLegalityError(
                f"ProgramShard reduction axis {option.axis!r} has no accumulator in loop {option.loop_nid}"
            )

    def _reduction_match(
        self, ir: KernelIR, option: ProgramShardOption, axis_blocks: frozenset[int], facts: _ShardFacts
    ) -> _ReductionMatch:
        """Resolve one additive accumulator, direct drain, and final store."""
        tensor = option.reduction_tensor
        if tensor is None:
            raise TransformLegalityError("reduction ProgramShard requires reduction_tensor")
        path = self._drain_store_path(ir, tensor, facts)
        if path is None:
            raise TransformLegalityError(f"ProgramShard reduction tensor {tensor!r} has no direct drain/store path")
        accumulator_leaf, drain_leaf, store_leaf = path
        if not self._is_additive_accumulator(ir, accumulator_leaf, option.axis, facts):
            raise TransformLegalityError(f"ProgramShard tensor {tensor!r} is not an additive {option.axis!r} reduction")
        accumulator_block = facts.owners[accumulator_leaf]
        if accumulator_block not in axis_blocks or facts.axis_loops[accumulator_block, option.axis] != option.loop_nid:
            raise TransformLegalityError("ProgramShard reduction tensor is not produced by the selected loop")
        drain_block = facts.owners[drain_leaf]
        drain = ir.tree.isa(drain_leaf)
        accumulator_region = drain.operand_bindings["src"]
        local_region = drain.operand_bindings["dst"]
        if accumulator_region.tensor != tensor or len(local_region.ranges) != 2:
            raise TransformLegalityError("ProgramShard requires a two-dimensional tensor-copy drain")
        if any(not isinstance(width, Const) for _lower, width in local_region.ranges):
            raise TransformLegalityError("ProgramShard requires statically sized exchange tiles")
        widths = tuple(width.value for _lower, width in local_region.ranges if isinstance(width, Const))
        for axis, width in zip(NKISendRecv.OPERAND_AXES["src"], widths):
            minimum = NKISendRecv.MIN_TILE_SIZE.get(axis)
            maximum = NKISendRecv.MAX_TILE_SIZE.get(axis)
            if (minimum is not None and width < minimum) or (maximum is not None and width > maximum):
                raise TransformLegalityError(
                    f"ProgramShard exchange axis {axis!r} width {width} is outside [{minimum}, {maximum}]"
                )
        if facts.buffers[local_region.tensor].location != "sbuf":
            raise TransformLegalityError("ProgramShard peer exchange requires an SBUF drain destination")
        store_block = facts.owners[store_leaf]
        if not facts.pipeline_overlap.isdisjoint({accumulator_leaf, drain_leaf, store_leaf, drain_block, store_block}):
            raise TransformLegalityError("ProgramShard reduction path cannot overlap an active software pipeline")
        return _ReductionMatch(
            axis_blocks, accumulator_leaf, drain_leaf, drain_block, store_leaf, accumulator_region, local_region
        )

    def _drain_store_path(self, ir: KernelIR, tensor: str, facts: _ShardFacts) -> tuple[int, int, int] | None:
        """Return a unique additive accumulator, tensor-copy drain, and store."""
        accumulators = [
            leaf
            for leaf in ir.dependency.touches_by_tensor.get(tensor, ())
            if tensor in ir.dependency.info(leaf).writes and facts.buffers[tensor].location == "psum"
        ]
        result: tuple[int, int, int] | None = None
        for accumulator in accumulators:
            consumers = ir.dependency.direct_consumers(accumulator)
            drains = [
                leaf
                for leaf in consumers
                if ir.tree.isa(leaf).op_cls is NKITensorCopy
                and ir.tree.isa(leaf).operand_bindings["src"].tensor == tensor
            ]
            if len(drains) != 1 or set(consumers) != set(drains):
                continue
            drain = drains[0]
            local = ir.tree.isa(drain).operand_bindings["dst"].tensor
            stores = [
                leaf
                for leaf in ir.dependency.direct_consumers(drain)
                if ir.tree.isa(leaf).op_cls is NKIStore and ir.tree.isa(leaf).operand_bindings["src"].tensor == local
            ]
            if len(stores) == 1 and set(ir.dependency.direct_consumers(drain)) == set(stores):
                candidate = (accumulator, drain, stores[0])
                result = candidate if result is None else None
                if result is None:
                    break
        return result

    def _is_additive_accumulator(self, ir: KernelIR, leaf_nid: int, axis: str, facts: _ShardFacts) -> bool:
        """Return whether one leaf additively reduces ``axis`` into PSUM."""
        leaf = ir.tree.isa(leaf_nid)
        block = ir.tree.block(facts.owners[leaf_nid])
        contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
        if not isinstance(contract, (BilinearReductionContract, ReductionContract)):
            return False
        reduction_axis = contract.reduction_axis
        region = leaf.operand_bindings.get(contract.output_operand)
        return bool(
            block.axis_map.get(reduction_axis) == axis
            and contract.combinator.combiner == "add"
            and region is not None
            and facts.buffers[region.tensor].location == "psum"
        )

    def _check_dependencies(
        self,
        ir: KernelIR,
        option: ProgramShardOption,
        axis_blocks: frozenset[int],
        match: _ReductionMatch | None,
        facts: _ShardFacts,
    ) -> None:
        """Reject a partial value consumed by an unsharded downstream loop."""
        allowed = None if match is None else (match.accumulator_leaf, match.drain_leaf)
        source_loop = ir.tree.loop(option.loop_nid)
        for producer, consumer in ir.dependency.graph.edges:
            producer_block = facts.owners[producer]
            consumer_block = facts.owners[consumer]
            if producer_block not in axis_blocks or consumer_block in axis_blocks or (producer, consumer) == allowed:
                continue
            tensor = ir.dependency.graph.edges[producer, consumer].get("tensor")
            writes = tuple(
                region
                for region in ir.dependency.info(producer).write_regions
                if isinstance(tensor, str) and region.tensor == tensor
            )
            consumer_loop = facts.axis_loops.get((consumer_block, option.axis))
            reads = tuple(
                region
                for region in ir.dependency.info(consumer).read_regions
                if isinstance(tensor, str) and region.tensor == tensor
            )
            source_indexed = any(
                source_loop.loop_var in to_affine(lower) for region in writes for lower, _ in region.ranges
            )
            consumer_indexed = consumer_loop is not None and any(
                ir.tree.loop(consumer_loop).loop_var in to_affine(lower)
                for region in reads
                for lower, _ in region.ranges
            )
            local_reuse = bool(
                isinstance(tensor, str)
                and consumer_loop is not None
                and not source_indexed
                and not consumer_indexed
                and source_loop.extent == ir.tree.loop(consumer_loop).extent
                and facts.buffers[tensor].location in {"sbuf", "psum"}
            )
            aligned = (
                consumer_loop is not None
                and facts.configured.get(consumer_loop) == option.programs
                and bool(writes)
                and bool(reads)
                and (
                    (
                        source_indexed
                        and consumer_indexed
                        and isinstance(tensor, str)
                        and self._same_program_partitions(
                            ir,
                            producer,
                            source_loop,
                            consumer,
                            ir.tree.loop(consumer_loop),
                            tensor,
                            option.programs,
                            facts,
                        )
                    )
                    or local_reuse
                )
            )
            if not aligned:
                raise TransformLegalityError(
                    f"ProgramShard loop {option.loop_nid} has shard-to-unsharded dependency {producer}->{consumer}"
                )

    def _same_program_partitions(
        self,
        ir: KernelIR,
        producer: int,
        producer_loop: ForNode,
        consumer: int,
        consumer_loop: ForNode,
        tensor: str,
        programs: int,
        facts: _ShardFacts,
    ) -> bool:
        """Return whether one dependency stays within each program's affine slice."""
        producer_info, consumer_info = ir.dependency.info(producer), ir.dependency.info(consumer)
        writes = tuple(region for region in producer_info.write_regions if region.tensor == tensor)
        reads = tuple(region for region in consumer_info.read_regions if region.tensor == tensor)
        buffer = facts.buffers[tensor]
        producer_views = tuple(
            _program_regions(writes, producer_loop, producer_info.extents, "producer", program, programs)
            for program in range(programs)
        )
        consumer_views = tuple(
            _program_regions(reads, consumer_loop, consumer_info.extents, "consumer", program, programs)
            for program in range(programs)
        )

        def overlaps(producer_program: int, consumer_program: int) -> bool:
            producer_regions, producer_extents = producer_views[producer_program]
            consumer_regions, consumer_extents = consumer_views[consumer_program]
            extents = {**producer_extents, **consumer_extents}
            return any(
                not regions_disjoint(write, read, buffer, buffer, extents)
                for write in producer_regions
                for read in consumer_regions
            )

        return all(
            overlaps(program, program)
            and all(not overlaps(other, program) for other in range(programs) if other != program)
            for program in range(programs)
        )

    def _check_hbm_writes(
        self,
        ir: KernelIR,
        option: ProgramShardOption,
        axis_blocks: frozenset[int],
        match: _ReductionMatch | None,
        facts: _ShardFacts,
    ) -> None:
        """Require every HBM write inside the selected loop to be disjoint."""
        loop_var = ir.tree.loop(option.loop_nid).loop_var
        for leaf_nid in ir.dependency.graph.nodes:
            block_nid = facts.owners[leaf_nid]
            if block_nid not in axis_blocks:
                continue
            for region in ir.dependency.info(leaf_nid).write_regions:
                if facts.buffers[region.tensor].location != "shared_hbm":
                    continue
                if match is not None:
                    raise TransformLegalityError("ProgramShard reduction loop cannot write a partial value to HBM")
                if not any(loop_var in to_affine(lower) for lower, _width in region.ranges):
                    raise TransformLegalityError(f"ProgramShard HBM write {region.tensor!r} is axis-invariant")

    def _set_shard_annotation(self, ir: KernelIR, option: ProgramShardOption) -> None:
        """Attach the selected materialized-loop program count to the root."""
        root = ir.tree.block(ir.tree.root)
        annotations = dict(root.annotations)
        shards = dict(annotations.get(PROGRAM_SHARDS_ANNOTATION, {}))
        if option.programs == 1:
            shards.pop(option.loop_nid)
        else:
            shards[option.loop_nid] = option.programs
        annotations[PROGRAM_SHARDS_ANNOTATION] = shards
        ir.tree.graph.nodes[ir.tree.root]["data"] = replace(root, annotations=annotations)

    def _insert_partial_combine(self, ir: KernelIR, option: ProgramShardOption, source_match: _ReductionMatch) -> None:
        """Exchange and add peer partials into a complete replicated result."""
        match = self._reduction_match(ir, option, source_match.axis_blocks, _shard_facts(ir))
        drain_block = ir.tree.block(match.drain_block)
        parent = ir.tree.parent(match.drain_leaf)
        if parent is None:
            raise AssertionError("ProgramShard drain leaf has no parent")
        local_buffer = ir.buffer(match.local_region.tensor)
        widths = tuple(width.value for _lower, width in match.local_region.ranges if isinstance(width, Const))
        peer = Buffer(
            name=fresh_name(ir, f"{match.local_region.tensor}_peer"),
            shape=widths,
            dtype=local_buffer.dtype,
            location="sbuf",
            storage_dtype=local_buffer.storage_dtype,
        )
        peer_region = BufferRegion(
            tensor=peer.name, ranges=tuple((Const(value=0), width) for _lower, width in match.local_region.ranges)
        )
        send_block = ir.tree.add_node(
            replace(drain_block, reads=(match.local_region,), writes=(peer_region,), alloc_buffers=())
        )
        send_leaf = ir.tree.add_node(
            ISANode(
                op_cls=NKISendRecv,
                operand_bindings={"src": match.local_region, "dst": peer_region},
                kwargs={"send_to_rank": "program_peer", "recv_from_rank": "program_peer", "pipe_id": 0},
            ),
            parent=send_block,
        )
        add_block = ir.tree.add_node(
            replace(
                drain_block,
                reads=(match.accumulator_region, peer_region),
                writes=(match.local_region,),
                alloc_buffers=(),
            )
        )
        ir.tree.add_node(
            ISANode(
                op_cls=NKITensorTensor,
                operand_bindings={"data1": match.accumulator_region, "data2": peer_region, "dst": match.local_region},
                kwargs={"op": "add"},
            ),
            parent=add_block,
        )
        owner = next(
            nid for nid in reversed((*ir.tree.ancestors(parent), parent)) if isinstance(ir.tree.data(nid), BlockNode)
        )
        owner_block = ir.tree.block(owner)
        ir.tree.graph.nodes[owner]["data"] = replace(owner_block, alloc_buffers=(*owner_block.alloc_buffers, peer))
        _replace_in_parent_children(ir.tree, parent, [match.drain_leaf], [match.drain_leaf, send_block, add_block])
        if ir.tree.parent(send_leaf) != send_block:
            raise AssertionError("ProgramShard exchange insertion failed")


def _shard_facts(ir: KernelIR) -> _ShardFacts:
    """Index immutable ownership, loop, buffer, and pipeline facts once."""
    block_nids = tuple(ir.tree.blocks())
    owners = {leaf_nid: owning_block(ir, leaf_nid) for leaf_nid in ir.dependency.graph.nodes}
    axes = {(block_nid, iter_var.axis) for block_nid in block_nids for iter_var in ir.tree.block(block_nid).iter_vars}
    axis_loops = {(block_nid, axis): axis_loop_for_block(ir, block_nid, axis) for block_nid, axis in axes}
    grouped: dict[tuple[int, str], list[int]] = {}
    for (block_nid, axis), loop_nid in axis_loops.items():
        if loop_nid is not None:
            grouped.setdefault((loop_nid, axis), []).append(block_nid)
    pipeline_depths: dict[int, int] = {}
    for block_nid in block_nids:
        annotation = ir.tree.block(block_nid).annotations.get("software_pipeline")
        if isinstance(annotation, dict):
            loop_nid = annotation.get("loop_nid")
            stages = annotation.get("stages")
            if isinstance(loop_nid, int) and isinstance(stages, tuple):
                pipeline_depths[loop_nid] = max(pipeline_depths.get(loop_nid, 0), max(stages))
    return _ShardFacts(
        block_nids=block_nids,
        blocks_by_loop_axis={key: tuple(values) for key, values in grouped.items()},
        owners=owners,
        axis_loops=axis_loops,
        buffers=ir.all_buffers(),
        configured=configured_program_shards(ir),
        pipeline_depths=pipeline_depths,
        pipeline_overlap=software_pipeline_overlap_nodes(ir),
    )


def _program_regions(
    regions: tuple[BufferRegion, ...], loop: ForNode, extents: dict[str, int], prefix: str, program: int, programs: int
) -> tuple[tuple[BufferRegion, ...], dict[str, int]]:
    """Return one program's regions with endpoint-local loop symbols."""
    local_extent = loop.extent // programs
    local_var = f"_{prefix}_program_local"
    substitutions: dict[str, Expr] = {name: Var(name=f"_{prefix}_{name}") for name in extents if name != loop.loop_var}
    substitutions[loop.loop_var] = Add(left=Var(name=local_var), right=Const(value=program * local_extent))
    renamed_extents = {f"_{prefix}_{name}": extent for name, extent in extents.items() if name != loop.loop_var}
    renamed_extents[local_var] = local_extent
    rewritten = tuple(
        BufferRegion(
            tensor=region.tensor,
            ranges=tuple(
                (substitute(lower, substitutions), substitute(width, substitutions)) for lower, width in region.ranges
            ),
        )
        for region in regions
    )
    return rewritten, renamed_extents


__all__ = ["ProgramShard", "ProgramShardOption"]
