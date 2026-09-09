"""Shard one output or additive-reduction axis across logical NeuronCores."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import Add, BlockNode, Buffer, BufferRegion, Const, Expr, ForNode, ISANode, KernelIR, Var, substitute
from nkigym.ir.arith.expr import expr_variables
from nkigym.ir.interval import regions_disjoint
from nkigym.ir.program_sharding import (
    PROGRAM_SHARDS_ANNOTATION,
    axis_loop_for_block,
    configured_program_shards,
    owning_block,
)
from nkigym.ops.base import (
    AxisRole,
    BilinearReductionContract,
    CopyContract,
    PeerExchangeContract,
    PointwiseContract,
    ReductionContract,
)
from nkigym.ops.sendrecv import NKISendRecv
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_tensor import NKITensorTensor
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    software_pipeline_overlap_nodes,
)
from nkigym.transforms.helper.canonical_rewrite import finalize_rewrite, fresh_name
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children

_PROGRAM_ID = "nl.program_id(0)"


@dataclass(frozen=True)
class ProgramShardOption(TransformOption):
    """Assign one loop to two programs, or remove one isolated assignment."""

    loop_nid: int
    axis: str
    programs: int
    reduction_tensor: str | None = None


@dataclass(frozen=True)
class _ReductionMatch:
    """Resolved additive partial, direct drain, and downstream consumers."""

    axis_blocks: frozenset[int]
    accumulator_leaf: int
    drain_leaf: int
    drain_block: int
    consumer_leaves: tuple[int, ...]
    accumulator_region: BufferRegion
    local_region: BufferRegion


@dataclass(frozen=True)
class _OwnedCombineMatch:
    """One additive combine loop matching an already owned store."""

    combine_leaf: int


@dataclass(frozen=True)
class _OwnedExchangeMatch:
    """One drain/exchange loop feeding an already owned combine."""

    drain_leaf: int
    exchange_leaf: int
    reader_operand: str


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
    """Partition disjoint output work or one complete additive reduction across cores.

    A reduction shard requires peer exchange and combination to reconstruct the
    complete value. Codegen permits the zero-peer fallback only after the shard
    annotation exists, so that communication is intrinsic to this one SPMD
    partition decision rather than a valid standalone rewrite.
    """

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
        """Apply one SPMD partition and its required reduction communication."""
        match = self._check_legality(ir, option, _shard_facts(ir))
        new_ir = copy_for_rewrite(ir)
        if isinstance(match, _OwnedCombineMatch):
            self._set_operation_ownership(new_ir, option, (match.combine_leaf,))
            finalize_rewrite(new_ir)
            return new_ir
        if isinstance(match, _OwnedExchangeMatch):
            self._apply_owned_exchange(new_ir, option, match)
            finalize_rewrite(new_ir)
            return new_ir
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

    def _check_legality(
        self, ir: KernelIR, option: ProgramShardOption, facts: _ShardFacts
    ) -> _ReductionMatch | _OwnedCombineMatch | _OwnedExchangeMatch | None:
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
        axis_blocks = frozenset(block_nids)
        ownership_match = None if reduction else self._ownership_match(ir, option, axis_blocks, facts)
        if (
            not reduction
            and not isinstance(ownership_match, _OwnedExchangeMatch)
            and any(
                ir.tree.isa(leaf_nid).op_cls is NKISendRecv
                for leaf_nid in ir.dependency.graph.nodes
                if facts.owners[leaf_nid] in block_nids
            )
        ):
            raise TransformLegalityError("ProgramShard cannot partition a loop containing a collective")
        self._check_pipeline_extent(ir, option, facts)
        if (reduction or ownership_match is not None) and option.loop_nid in facts.pipeline_overlap:
            raise TransformLegalityError("owned or reduction ProgramShard cannot overlap an active software pipeline")
        if ownership_match is not None:
            return ownership_match
        match = self._reduction_match(ir, option, axis_blocks, facts) if reduction else None
        self._check_dependencies(ir, option, axis_blocks, match, facts)
        self._check_hbm_writes(ir, option, axis_blocks, match, facts)
        return match

    def _check_unshard(self, ir: KernelIR, option: ProgramShardOption, facts: _ShardFacts) -> None:
        """Require one isolated output shard with no collective side effects."""
        if facts.configured != {option.loop_nid: 2} or option.reduction_tensor is not None:
            raise TransformLegalityError("ProgramShard can remove only one isolated independent shard")
        if any(_uses_direct_program_index(ir, leaf_nid) for leaf_nid in ir.dependency.graph.nodes):
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
        """Resolve one additive accumulator and its direct SBUF drain."""
        tensor = option.reduction_tensor
        if tensor is None:
            raise TransformLegalityError("reduction ProgramShard requires reduction_tensor")
        path = self._direct_drain_path(ir, tensor, facts)
        if path is None:
            raise TransformLegalityError(f"ProgramShard reduction tensor {tensor!r} has no unique direct drain path")
        accumulator_leaf, drain_leaf, consumer_leaves = path
        if not self._is_additive_accumulator(ir, accumulator_leaf, option.axis, facts):
            raise TransformLegalityError(f"ProgramShard tensor {tensor!r} is not an additive {option.axis!r} reduction")
        accumulator_block = facts.owners[accumulator_leaf]
        if accumulator_block not in axis_blocks or facts.axis_loops[accumulator_block, option.axis] != option.loop_nid:
            raise TransformLegalityError("ProgramShard reduction tensor is not produced by the selected loop")
        drain_block = facts.owners[drain_leaf]
        drain = ir.tree.isa(drain_leaf)
        drain_loop = ir.tree.parent(drain_leaf)
        if (
            drain_loop is None
            or not isinstance(ir.tree.data(drain_loop), ForNode)
            or ir.tree.children(drain_loop) != [drain_leaf]
        ):
            raise TransformLegalityError("ProgramShard reduction drain must be the sole operation in one tile loop")
        if facts.configured.keys() & set(ir.tree.ancestors(drain_leaf)):
            raise TransformLegalityError("reduction ProgramShard requires a replicated drain path")
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
        path_nodes = {
            accumulator_leaf,
            drain_leaf,
            drain_block,
            *consumer_leaves,
            *(facts.owners[leaf] for leaf in consumer_leaves),
        }
        if not facts.pipeline_overlap.isdisjoint(path_nodes):
            raise TransformLegalityError("ProgramShard reduction path cannot overlap an active software pipeline")
        return _ReductionMatch(
            axis_blocks, accumulator_leaf, drain_leaf, drain_block, consumer_leaves, accumulator_region, local_region
        )

    def _direct_drain_path(
        self, ir: KernelIR, tensor: str, facts: _ShardFacts
    ) -> tuple[int, int, tuple[int, ...]] | None:
        """Return a unique accumulator, tensor-copy drain, and its consumers."""
        accumulators = [
            leaf
            for leaf in ir.dependency.touches_by_tensor.get(tensor, ())
            if tensor in ir.dependency.info(leaf).writes and facts.buffers[tensor].location == "psum"
        ]
        result: tuple[int, int, tuple[int, ...]] | None = None
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
            downstream = tuple(ir.dependency.direct_consumers(drain))
            if downstream:
                candidate = (accumulator, drain, downstream)
                result = candidate if result is None else None
                if result is None:
                    break
        return result

    def _ownership_match(
        self, ir: KernelIR, option: ProgramShardOption, axis_blocks: frozenset[int], facts: _ShardFacts
    ) -> _OwnedCombineMatch | _OwnedExchangeMatch | None:
        """Resolve one atomic propagation of an existing store ownership decision."""
        match: _OwnedCombineMatch | _OwnedExchangeMatch | None = None
        if set(facts.configured.values()) == {option.programs}:
            match = self._owned_combine_match(ir, option, axis_blocks, facts)
            if match is None:
                match = self._owned_exchange_match(ir, option, axis_blocks, facts)
        return match

    def _owned_combine_match(
        self, ir: KernelIR, option: ProgramShardOption, axis_blocks: frozenset[int], facts: _ShardFacts
    ) -> _OwnedCombineMatch | None:
        """Match one additive producer whose direct store already owns this axis."""
        leaves = tuple(leaf for leaf in ir.dependency.graph.nodes if facts.owners[leaf] in axis_blocks)
        match: _OwnedCombineMatch | None = None
        if len(leaves) == 1:
            combine_leaf = leaves[0]
            combine = ir.tree.isa(combine_leaf)
            contract = combine.op_cls.algebraic_contract(combine.kwargs)
            output = (
                combine.operand_bindings.get(contract.output_operand)
                if isinstance(contract, PointwiseContract) and contract.operator == "add"
                else None
            )
            owned_stores = [
                consumer
                for consumer in ir.dependency.direct_consumers(combine_leaf)
                if _owned_store_consumes(ir, combine_leaf, consumer, output, option)
            ]
            if (
                output is not None
                and "program_ownership" not in combine.kwargs
                and len(owned_stores) == 1
                and set(ir.dependency.direct_consumers(combine_leaf)) == set(owned_stores)
            ):
                match = _OwnedCombineMatch(combine_leaf)
        return match

    def _owned_exchange_match(
        self, ir: KernelIR, option: ProgramShardOption, axis_blocks: frozenset[int], facts: _ShardFacts
    ) -> _OwnedExchangeMatch | None:
        """Match one copy plus peer exchange feeding an already owned additive combine."""
        leaves = tuple(leaf for leaf in ir.dependency.graph.nodes if facts.owners[leaf] in axis_blocks)
        copies = tuple(
            leaf
            for leaf in leaves
            if isinstance(ir.tree.isa(leaf).op_cls.algebraic_contract(ir.tree.isa(leaf).kwargs), CopyContract)
        )
        exchanges = tuple(
            leaf
            for leaf in leaves
            if isinstance(ir.tree.isa(leaf).op_cls.algebraic_contract(ir.tree.isa(leaf).kwargs), PeerExchangeContract)
        )
        match: _OwnedExchangeMatch | None = None
        if len(copies) == len(exchanges) == 1 and set(leaves) == {copies[0], exchanges[0]}:
            drain_leaf, exchange_leaf = copies[0], exchanges[0]
            drain, exchange = ir.tree.isa(drain_leaf), ir.tree.isa(exchange_leaf)
            drain_contract = drain.op_cls.algebraic_contract(drain.kwargs)
            exchange_contract = exchange.op_cls.algebraic_contract(exchange.kwargs)
            if not isinstance(drain_contract, CopyContract) or not isinstance(exchange_contract, PeerExchangeContract):
                raise AssertionError("owned exchange contracts changed during matching")
            drain_input = drain.operand_bindings.get(drain_contract.input_operand)
            drain_output = drain.operand_bindings.get(drain_contract.output_operand)
            exchange_input = exchange.operand_bindings.get(exchange_contract.input_operand)
            exchange_output = exchange.operand_bindings.get(exchange_contract.output_operand)
            consumers = ir.dependency.direct_consumers(exchange_leaf)
            owned_combines = [
                consumer
                for consumer in consumers
                if _owned_combine_consumes(
                    ir, exchange_leaf, exchange_output, drain_leaf, drain_input, consumer, option
                )
            ]
            if (
                drain_input is not None
                and drain_output == exchange_input
                and exchange_output is not None
                and "program_ownership" not in drain.kwargs
                and "program_ownership" not in exchange.kwargs
                and exchange_leaf in ir.dependency.direct_consumers(drain_leaf)
                and len(owned_combines) == 1
                and set(consumers) == set(owned_combines)
            ):
                match = _OwnedExchangeMatch(drain_leaf, exchange_leaf, drain_contract.input_operand)
        return match

    def _set_operation_ownership(self, ir: KernelIR, option: ProgramShardOption, leaf_nids: tuple[int, ...]) -> None:
        """Predicate one already-partitioned operation path by the selected output axis."""
        ownership = (option.axis, option.programs)
        for leaf_nid in leaf_nids:
            leaf = ir.tree.isa(leaf_nid)
            ir.tree.graph.nodes[leaf_nid]["data"] = replace(
                leaf, kwargs={**leaf.kwargs, "program_ownership": ownership}
            )

    def _apply_owned_exchange(self, ir: KernelIR, option: ProgramShardOption, match: _OwnedExchangeMatch) -> None:
        """Route peer-owned partials while pruning the matching drain/exchange loop."""
        loop_var = ir.tree.loop(option.loop_nid).loop_var
        peer_offset = Var(name="(nl.num_programs(0) - 1) * (1 - 2 * nl.program_id(0))")
        peer_iteration = Add(left=Var(name=loop_var), right=peer_offset)
        drain = ir.tree.isa(match.drain_leaf)
        old_source = drain.operand_bindings[match.reader_operand]
        peer_source = BufferRegion(
            tensor=old_source.tensor,
            ranges=tuple(
                (substitute(lower, {loop_var: peer_iteration}), substitute(width, {loop_var: peer_iteration}))
                for lower, width in old_source.ranges
            ),
        )
        bindings = dict(drain.operand_bindings)
        bindings[match.reader_operand] = peer_source
        ir.tree.graph.nodes[match.drain_leaf]["data"] = replace(drain, operand_bindings=bindings)
        drain_block_nid = owning_block(ir, match.drain_leaf)
        drain_block = ir.tree.block(drain_block_nid)
        if old_source not in drain_block.reads:
            raise AssertionError("owned exchange source is absent from its block")
        reads = tuple(peer_source if region == old_source else region for region in drain_block.reads)
        ir.tree.graph.nodes[drain_block_nid]["data"] = replace(drain_block, reads=reads)
        self._set_operation_ownership(ir, option, (match.drain_leaf, match.exchange_leaf))

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
            reads = tuple(
                region
                for region in ir.dependency.info(consumer).read_regions
                if isinstance(tensor, str) and region.tensor == tensor
            )
            source_indexed = any(
                source_loop.loop_var in expr_variables(lower) for region in writes for lower, _ in region.ranges
            )
            direct_program_indexed = any(
                _PROGRAM_ID in expr_variables(lower) for region in reads for lower, _ in region.ranges
            )
            same_axis_loop = facts.axis_loops.get((consumer_block, option.axis))
            consumer_loops = sorted(
                {
                    loop_nid
                    for (block_nid, _axis), loop_nid in facts.axis_loops.items()
                    if block_nid == consumer_block
                    and loop_nid is not None
                    and facts.configured.get(loop_nid) == option.programs
                }
            )
            aligned = bool(
                source_indexed
                and direct_program_indexed
                and writes
                and reads
                and isinstance(tensor, str)
                and self._same_direct_program_partitions(
                    ir, producer, source_loop, consumer, tensor, option.programs, facts
                )
            )
            for consumer_loop in consumer_loops:
                if aligned:
                    break
                loop = ir.tree.loop(consumer_loop)
                consumer_indexed = any(
                    loop.loop_var in expr_variables(lower) for region in reads for lower, _ in region.ranges
                )
                local_reuse = bool(
                    isinstance(tensor, str)
                    and consumer_loop == same_axis_loop
                    and not source_indexed
                    and not consumer_indexed
                    and source_loop.extent == loop.extent
                    and facts.buffers[tensor].location in {"sbuf", "psum"}
                )
                aligned = local_reuse or bool(
                    source_indexed
                    and consumer_indexed
                    and writes
                    and reads
                    and isinstance(tensor, str)
                    and self._same_program_partitions(
                        ir, producer, source_loop, consumer, loop, tensor, option.programs, facts
                    )
                )
                if aligned:
                    break
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
        producer_extents = _endpoint_loop_extents(ir, producer, producer_info.extents)
        consumer_extents = _endpoint_loop_extents(ir, consumer, consumer_info.extents)
        producer_views = tuple(
            _program_regions(writes, producer_loop, producer_extents, "producer", program, programs)
            for program in range(programs)
        )
        consumer_views = tuple(
            _program_regions(reads, consumer_loop, consumer_extents, "consumer", program, programs)
            for program in range(programs)
        )
        return _same_program_views(buffer, producer_views, consumer_views, programs)

    def _same_direct_program_partitions(
        self,
        ir: KernelIR,
        producer: int,
        producer_loop: ForNode,
        consumer: int,
        tensor: str,
        programs: int,
        facts: _ShardFacts,
    ) -> bool:
        """Return whether a loop shard matches direct program-indexed consumers."""
        producer_info, consumer_info = ir.dependency.info(producer), ir.dependency.info(consumer)
        writes = tuple(region for region in producer_info.write_regions if region.tensor == tensor)
        reads = tuple(region for region in consumer_info.read_regions if region.tensor == tensor)
        producer_extents = _endpoint_loop_extents(ir, producer, producer_info.extents)
        consumer_extents = _endpoint_loop_extents(ir, consumer, consumer_info.extents)
        producer_views = tuple(
            _program_regions(writes, producer_loop, producer_extents, "producer", program, programs)
            for program in range(programs)
        )
        consumer_views = tuple(
            _direct_program_regions(reads, consumer_extents, "consumer", program) for program in range(programs)
        )
        return _same_program_views(facts.buffers[tensor], producer_views, consumer_views, programs)

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
            if "program_ownership" in ir.tree.isa(leaf_nid).kwargs:
                raise TransformLegalityError("ProgramShard cannot partition an already owned store")
            for region in ir.dependency.info(leaf_nid).write_regions:
                if facts.buffers[region.tensor].location != "shared_hbm":
                    continue
                if match is not None:
                    raise TransformLegalityError("ProgramShard reduction loop cannot write a partial value to HBM")
                if any(_PROGRAM_ID in expr_variables(lower) for lower, _width in region.ranges):
                    raise TransformLegalityError("ProgramShard cannot partition an already program-owned HBM write")
                if not any(loop_var in expr_variables(lower) for lower, _width in region.ranges):
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
        drain_loop_nid = ir.tree.parent(match.drain_leaf)
        if drain_loop_nid is None:
            raise AssertionError("ProgramShard drain leaf has no parent")
        sequence_parent = ir.tree.parent(drain_loop_nid)
        if sequence_parent is None:
            raise AssertionError("ProgramShard drain loop has no parent")
        drain_loop = ir.tree.loop(drain_loop_nid)
        local_buffer = ir.buffer(match.local_region.tensor)
        accumulator_dtype = ir.buffer(match.accumulator_region.tensor).physical_dtype()
        partial_region = match.local_region
        extra_buffers: tuple[Buffer, ...] = ()
        if local_buffer.physical_dtype() != accumulator_dtype:
            partial = replace(
                local_buffer,
                name=fresh_name(ir, f"{match.local_region.tensor}_partial"),
                dtype=accumulator_dtype,
                storage_dtype=None,
            )
            partial_region = replace(match.local_region, tensor=partial.name)
            drain_leaf = ir.tree.isa(match.drain_leaf)
            drain_bindings = dict(drain_leaf.operand_bindings)
            drain_bindings["dst"] = partial_region
            ir.tree.graph.nodes[match.drain_leaf]["data"] = replace(drain_leaf, operand_bindings=drain_bindings)
            drain_block = replace(drain_block, writes=(partial_region,))
            extra_buffers = (partial,)
        partial_buffer = local_buffer if not extra_buffers else extra_buffers[0]
        peer = replace(partial_buffer, name=fresh_name(ir, f"{match.local_region.tensor}_peer"))
        peer_region = replace(partial_region, tensor=peer.name)
        send_block = ir.tree.add_node(
            replace(drain_block, reads=(partial_region,), writes=(peer_region,), alloc_buffers=(), annotations={})
        )
        send_loop = ir.tree.add_node(drain_loop, parent=send_block)
        send_leaf = ir.tree.add_node(
            ISANode(
                op_cls=NKISendRecv,
                operand_bindings={"src": partial_region, "dst": peer_region},
                kwargs={"send_to_rank": "program_peer", "recv_from_rank": "program_peer", "pipe_id": 0},
            ),
            parent=send_loop,
        )
        add_block = ir.tree.add_node(
            replace(
                drain_block,
                reads=(match.accumulator_region, peer_region),
                writes=(match.local_region,),
                alloc_buffers=(),
                annotations={},
            )
        )
        add_loop = ir.tree.add_node(drain_loop, parent=add_block)
        ir.tree.add_node(
            ISANode(
                op_cls=NKITensorTensor,
                operand_bindings={"data1": match.accumulator_region, "data2": peer_region, "dst": match.local_region},
                kwargs={"op": "add"},
            ),
            parent=add_loop,
        )
        ir.tree.graph.nodes[match.drain_block]["data"] = replace(
            drain_block, alloc_buffers=(*drain_block.alloc_buffers, *extra_buffers, peer)
        )
        _replace_in_parent_children(ir.tree, sequence_parent, [drain_loop_nid], [drain_loop_nid, send_block, add_block])
        if ir.tree.parent(send_leaf) != send_loop:
            raise AssertionError("ProgramShard exchange insertion failed")


def _owned_store_consumes(
    ir: KernelIR, source_leaf: int, leaf_nid: int, source: BufferRegion | None, option: ProgramShardOption
) -> bool:
    """Return whether one copy-like HBM store owns the selected output axis."""
    if source is None:
        return False
    store = ir.tree.isa(leaf_nid)
    contract = store.op_cls.algebraic_contract(store.kwargs)
    destination = store.operand_bindings.get(contract.output_operand) if isinstance(contract, CopyContract) else None
    return bool(
        isinstance(contract, CopyContract)
        and _regions_align_on_axis(
            ir, source_leaf, source, leaf_nid, store.operand_bindings.get(contract.input_operand), option.axis
        )
        and destination is not None
        and ir.buffer(destination.tensor).location == "shared_hbm"
        and store.kwargs.get("program_ownership") == (option.axis, option.programs)
    )


def _owned_combine_consumes(
    ir: KernelIR,
    peer_leaf: int,
    peer_region: BufferRegion | None,
    local_leaf: int,
    local_region: BufferRegion | None,
    combine_leaf: int,
    option: ProgramShardOption,
) -> bool:
    """Return whether one owned additive combine joins the local and peer values."""
    if peer_region is None or local_region is None:
        return False
    combine = ir.tree.isa(combine_leaf)
    contract = combine.op_cls.algebraic_contract(combine.kwargs)
    if (
        not isinstance(contract, PointwiseContract)
        or contract.operator != "add"
        or len(contract.input_operands) != 2
        or combine.kwargs.get("program_ownership") != (option.axis, option.programs)
    ):
        return False
    inputs = tuple(combine.operand_bindings.get(operand) for operand in contract.input_operands)
    output = combine.operand_bindings.get(contract.output_operand)
    stores = [
        consumer
        for consumer in ir.dependency.direct_consumers(combine_leaf)
        if _owned_store_consumes(ir, combine_leaf, consumer, output, option)
    ]
    peer_inputs = [
        region
        for region in inputs
        if _regions_align_on_axis(ir, peer_leaf, peer_region, combine_leaf, region, option.axis)
    ]
    local_inputs = [
        region
        for region in inputs
        if _regions_align_on_axis(ir, local_leaf, local_region, combine_leaf, region, option.axis)
    ]
    return len(peer_inputs) == len(local_inputs) == len(stores) == 1 and peer_inputs[0] != local_inputs[0]


def _regions_align_on_axis(
    ir: KernelIR, first_leaf: int, first: BufferRegion | None, second_leaf: int, second: BufferRegion | None, axis: str
) -> bool:
    """Return whether two regions cover the same coordinates under their axis loops."""
    if first is None or second is None or first.tensor != second.tensor or len(first.ranges) != len(second.ranges):
        return False
    first_loop_nid = axis_loop_for_block(ir, owning_block(ir, first_leaf), axis)
    second_loop_nid = axis_loop_for_block(ir, owning_block(ir, second_leaf), axis)
    if first_loop_nid is None or second_loop_nid is None:
        return False
    first_loop, second_loop = ir.tree.loop(first_loop_nid), ir.tree.loop(second_loop_nid)
    common = Var(name="_program_owned_axis")
    first_ranges = tuple(
        (substitute(lower, {first_loop.loop_var: common}), substitute(width, {first_loop.loop_var: common}))
        for lower, width in first.ranges
    )
    second_ranges = tuple(
        (substitute(lower, {second_loop.loop_var: common}), substitute(width, {second_loop.loop_var: common}))
        for lower, width in second.ranges
    )
    return first_loop.extent == second_loop.extent and first_ranges == second_ranges


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


def _direct_program_regions(
    regions: tuple[BufferRegion, ...], extents: dict[str, int], prefix: str, program: int
) -> tuple[tuple[BufferRegion, ...], dict[str, int]]:
    """Return direct program-indexed regions with endpoint-local symbols."""
    substitutions: dict[str, Expr] = {name: Var(name=f"_{prefix}_{name}") for name in extents}
    substitutions[_PROGRAM_ID] = Const(value=program)
    renamed_extents = {f"_{prefix}_{name}": extent for name, extent in extents.items()}
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


def _same_program_views(
    buffer: Buffer,
    producer_views: tuple[tuple[tuple[BufferRegion, ...], dict[str, int]], ...],
    consumer_views: tuple[tuple[tuple[BufferRegion, ...], dict[str, int]], ...],
    programs: int,
) -> bool:
    """Return whether two region families have identical program ownership."""

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


def _endpoint_loop_extents(ir: KernelIR, leaf_nid: int, extents: dict[str, int]) -> dict[str, int]:
    """Return dependency and enclosing-loop bounds for one shard endpoint."""
    result = dict(extents)
    result.update(
        {
            loop.loop_var: loop.extent
            for nid in ir.tree.ancestors(leaf_nid)
            if isinstance((loop := ir.tree.data(nid)), ForNode)
        }
    )
    return result


def _uses_direct_program_index(ir: KernelIR, leaf_nid: int) -> bool:
    """Return whether one operation retains direct program-local state."""
    leaf = ir.tree.isa(leaf_nid)
    info = ir.dependency.info(leaf_nid)
    expressions = (
        value for region in (*info.read_regions, *info.write_regions) for bounds in region.ranges for value in bounds
    )
    return (
        leaf.op_cls is NKISendRecv
        or "program_ownership" in leaf.kwargs
        or any(_PROGRAM_ID in expr_variables(value) for value in expressions)
        or any(_PROGRAM_ID in expr_variables(pattern.offset) for pattern in leaf.access_patterns.values())
    )


__all__ = ["ProgramShard", "ProgramShardOption"]
