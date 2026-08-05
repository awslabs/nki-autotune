"""Contract-driven dataflow analysis for online fusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Expr, Var, substitute
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode
from nkigym.ops.base import (
    BilinearReductionContract,
    CopyContract,
    InitializerContract,
    OperatorContract,
    PermutationContract,
    PointwiseContract,
    ReductionContract,
)
from nkigym.transforms._canonical_rewrite import block_chain, is_canonical_block, owning_block, required_spec
from nkigym.transforms._online_fusion_algebra import AlgebraEvaluation, evaluate_algebra
from nkigym.transforms._online_fusion_types import (
    BinaryFactor,
    ConstantFactor,
    DeferredFactor,
    FactorExpression,
    OnlineFusionMatch,
    OnlineFusionStage,
    StateFactor,
    UnaryFactor,
    ValueGraph,
    contract_input_operands,
    contract_output_operand,
    contract_output_operands,
    factor_states,
)


def build_value_graph(ir: KernelIR) -> ValueGraph:
    """Derive semantic SSA use-def edges from canonical ISA leaves."""
    leaves = tuple(nid for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode))
    contracts: dict[int, OperatorContract] = {}
    output_by_leaf: dict[int, str] = {}
    inputs_by_leaf: dict[int, dict[str, str]] = {}
    predecessors: dict[int, dict[str, int | None]] = {}
    successors_mut: dict[int, list[int]] = {leaf: [] for leaf in leaves}
    tensor_axes: dict[str, tuple[str, ...]] = {}
    initializers_mut: dict[str, list[int]] = {}
    current_producer: dict[str, int] = {}

    for leaf_nid in leaves:
        leaf = ir.tree.isa(leaf_nid)
        contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
        if contract is None:
            raise ValueError(f"{leaf.op_cls.__name__} has no algebraic contract")
        contracts[leaf_nid] = contract
        block = ir.tree.block(owning_block(ir.tree, leaf_nid))
        _record_tensor_axes(leaf, block.axis_map, tensor_axes)
        semantic_inputs = contract_input_operands(contract)
        bound_inputs = {
            slot: leaf.operand_bindings[slot].tensor for slot in semantic_inputs if slot in leaf.operand_bindings
        }
        inputs_by_leaf[leaf_nid] = bound_inputs
        predecessors[leaf_nid] = {}
        for slot, tensor in bound_inputs.items():
            producer = current_producer.get(tensor)
            predecessors[leaf_nid][slot] = producer
            if producer is not None:
                successors_mut[producer].append(leaf_nid)
        output_operand = contract_output_operand(contract)
        if output_operand not in leaf.operand_bindings:
            raise ValueError(f"{leaf.op_cls.__name__} contract output {output_operand!r} is unbound")
        output_tensor = leaf.operand_bindings[output_operand].tensor
        output_by_leaf[leaf_nid] = output_tensor
        for produced_operand in contract_output_operands(contract):
            if produced_operand not in leaf.operand_bindings:
                raise ValueError(f"{leaf.op_cls.__name__} contract output {produced_operand!r} is unbound")
            produced_tensor = leaf.operand_bindings[produced_operand].tensor
            current_producer[produced_tensor] = leaf_nid
        if isinstance(contract, InitializerContract):
            initializers_mut.setdefault(output_tensor, []).append(leaf_nid)

    successors = {leaf: tuple(dict.fromkeys(consumers)) for leaf, consumers in successors_mut.items()}
    initializers = {tensor: tuple(items) for tensor, items in initializers_mut.items()}
    return ValueGraph(
        leaves=leaves,
        contracts=contracts,
        output_by_leaf=output_by_leaf,
        input_tensors_by_leaf=inputs_by_leaf,
        predecessors=predecessors,
        successors=successors,
        tensor_axes=tensor_axes,
        initializers_by_tensor=initializers,
    )


def detect_online_fusion(ir: KernelIR) -> list[OnlineFusionMatch]:
    """Return the next independently selectable fusion stage."""
    complete = detect_complete_online_fusion(ir)
    matches: list[OnlineFusionMatch] = []
    for match in complete:
        if len(match.stages) > 2:
            graph = build_value_graph(ir)
            evaluation = evaluate_algebra(ir, graph, match.progress_axis)
            prefix = _build_incremental_prefix(ir, graph, match.progress_axis, evaluation, 2)
            if prefix is not None:
                chunk_sizes = tuple(size for size in prefix.chunk_sizes if size in match.chunk_sizes)
                if chunk_sizes:
                    matches.append(replace(prefix, chunk_sizes=chunk_sizes))
        else:
            matches.append(match)
    return matches


def detect_complete_online_fusion(ir: KernelIR) -> list[OnlineFusionMatch]:
    """Return maximal canonical chains proven by the registered contracts."""
    matches: list[OnlineFusionMatch] = []
    compatible_axes = tuple(
        progress_axis
        for progress_axis in _candidate_progress_axes(ir)
        if all(
            block_nid == ir.tree.root or _is_online_fusion_block(ir, block_nid, progress_axis)
            for block_nid in ir.tree.blocks()
        )
    )
    if compatible_axes:
        graph = build_value_graph(ir)
        for progress_axis in compatible_axes:
            evaluation = evaluate_algebra(ir, graph, progress_axis)
            if len(evaluation.stages) >= 2:
                match = _build_match(ir, graph, progress_axis, evaluation)
                if match is not None:
                    matches.append(match)
    return matches


def _is_online_fusion_block(ir: KernelIR, block_nid: int, progress_axis: str) -> bool:
    """Accept canonical blocks and exact non-progress outer-loop factorizations."""
    valid = is_canonical_block(ir, block_nid)
    chain = block_chain(ir.tree, block_nid)
    if not valid and chain is not None:
        block = ir.tree.block(block_nid)
        leaf = chain[-1]
        if not isinstance(leaf, ISANode):
            raise AssertionError(f"canonical block chain {block_nid} has no ISA leaf")
        operand_names = {slot: region.tensor for slot, region in leaf.operand_bindings.items()}
        spec = required_spec(ir, leaf.op_cls, operand_names, block.axis_map, leaf.kwargs)
        substitutions = _canonical_iter_substitutions(spec.block, block)
        valid = substitutions is not None and _factored_loops_match(
            spec.loops, tuple(payload for payload in chain[1:-1] if isinstance(payload, ForNode)), progress_axis
        )
        if valid and substitutions is not None:
            expected_block = replace(
                spec.block,
                iter_values=block.iter_values,
                reads=tuple(_substitute_region(region, substitutions) for region in spec.block.reads),
                writes=tuple(_substitute_region(region, substitutions) for region in spec.block.writes),
            )
            expected_leaf = replace(
                spec.leaf,
                operand_bindings={
                    slot: _substitute_region(region, substitutions)
                    for slot, region in spec.leaf.operand_bindings.items()
                },
            )
            valid = replace(block, alloc_buffers=()) == expected_block and leaf == expected_leaf
    return valid


def _canonical_iter_substitutions(canonical: BlockNode, actual: BlockNode) -> dict[str, Expr] | None:
    """Map canonical loop variables to one factored block's linearized values."""
    substitutions: dict[str, Expr] = {}
    valid = canonical.iter_vars == actual.iter_vars and canonical.axis_map == actual.axis_map
    if valid:
        for expected, replacement in zip(canonical.iter_values, actual.iter_values):
            if isinstance(expected, Var):
                substitutions[expected.name] = replacement
            elif expected != replacement:
                valid = False
                break
    return substitutions if valid else None


def _factored_loops_match(canonical: tuple[ForNode, ...], actual: tuple[ForNode, ...], progress_axis: str) -> bool:
    """Check ordered exact loop products while forbidding progress-axis factorization."""
    cursor = 0
    valid = True
    for expected in canonical:
        axis = _loop_axis(expected.loop_var)
        group: list[ForNode] = []
        while cursor < len(actual) and _loop_axis(actual[cursor].loop_var) == axis:
            group.append(actual[cursor])
            cursor += 1
        product = 1
        for loop in group:
            product *= loop.extent
        names = [loop.loop_var for loop in group]
        dense_names = [f"i_{axis}_{index}" for index in range(len(group))]
        if not group or product != expected.extent or names != dense_names:
            valid = False
            break
        if axis == progress_axis and group != [expected]:
            valid = False
            break
    return valid and cursor == len(actual)


def _loop_axis(loop_var: str) -> str:
    """Return the concrete axis encoded in one normalized loop variable."""
    body = loop_var[2:] if loop_var.startswith("i_") else loop_var
    return body.rsplit("_", 1)[0]


def _substitute_region(region: BufferRegion, substitutions: dict[str, Expr]) -> BufferRegion:
    """Apply canonical-to-factored iteration substitutions to one region."""
    return replace(
        region,
        ranges=tuple(
            (substitute(lower, substitutions), substitute(width, substitutions)) for lower, width in region.ranges
        ),
    )


def _build_incremental_prefix(
    ir: KernelIR, graph: ValueGraph, progress_axis: str, evaluation: AlgebraEvaluation, stage_count: int
) -> OnlineFusionMatch | None:
    """Build a recurrence prefix retained beside the original materialized path."""
    stages = evaluation.stages[:stage_count]
    stage_leaves = {stage.reducer_leaf for stage in stages}
    relevant = _ancestors(graph, stage_leaves)
    absorbed = {
        leaf_nid
        for leaf_nid in relevant
        if leaf_nid in stage_leaves
        or evaluation.values_by_leaf[leaf_nid].depends_on_progress
        or bool(factor_states(evaluation.values_by_leaf[leaf_nid].factor))
    }
    for leaf_nid in tuple(absorbed):
        output = graph.output_by_leaf[leaf_nid]
        absorbed.update(graph.initializers_by_tensor.get(output, ()))

    external_inputs: list[str] = []
    for leaf_nid in graph.leaves:
        if leaf_nid not in absorbed:
            continue
        for slot, tensor in graph.input_tensors_by_leaf[leaf_nid].items():
            producer = graph.predecessors[leaf_nid][slot]
            if producer not in absorbed and tensor not in external_inputs:
                external_inputs.append(tensor)

    chunk_sizes = _chunk_sizes(ir, absorbed, progress_axis)
    result: OnlineFusionMatch | None = None
    if len(stages) == stage_count and chunk_sizes:
        order = {leaf: index for index, leaf in enumerate(graph.leaves)}
        derivation = tuple(sorted(absorbed, key=order.__getitem__))
        result = OnlineFusionMatch(
            progress_axis=progress_axis,
            progress_extent=ir.axis_extent(progress_axis),
            stages=stages,
            derivation_leaves=derivation,
            absorbed_blocks=tuple(owning_block(ir.tree, leaf) for leaf in derivation),
            external_inputs=tuple(external_inputs),
            external_outputs=(stages[-1].state_tensor,),
            chunk_sizes=chunk_sizes,
            deferred_factor=None,
            incremental_prefix=True,
        )
    return result


def _record_tensor_axes(leaf: ISANode, axis_map: Mapping[str, str], result: dict[str, tuple[str, ...]]) -> None:
    """Record and validate each bound tensor's concrete logical axes."""
    for slot, region in leaf.operand_bindings.items():
        abstract_axes = leaf.op_cls.OPERAND_AXES[slot]
        concrete_axes = tuple(axis_map[axis] for axis in abstract_axes if axis in axis_map)
        prior = result.get(region.tensor)
        if prior is not None and prior != concrete_axes:
            raise ValueError(f"tensor {region.tensor!r} has inconsistent axes {prior} and {concrete_axes}")
        result[region.tensor] = concrete_axes


def _candidate_progress_axes(ir: KernelIR) -> tuple[str, ...]:
    """Return concrete axes used by at least one associative reduction."""
    axes: set[str] = set()
    for leaf_nid in ir.tree.preorder():
        node = ir.tree.data(leaf_nid)
        if not isinstance(node, ISANode):
            continue
        contract = node.op_cls.algebraic_contract(node.kwargs)
        if isinstance(contract, (ReductionContract, BilinearReductionContract)):
            block = ir.tree.block(owning_block(ir.tree, leaf_nid))
            axes.add(block.axis_map[contract.reduction_axis])
    return tuple(sorted(axes))


def _build_match(
    ir: KernelIR, graph: ValueGraph, progress_axis: str, evaluation: AlgebraEvaluation
) -> OnlineFusionMatch | None:
    """Compute the exact absorbed subgraph and external boundary."""
    stage_leaves = {stage.reducer_leaf for stage in evaluation.stages}
    relevant = _ancestors(graph, stage_leaves)
    absorbed: set[int] = set()
    for leaf_nid in relevant:
        value = evaluation.values_by_leaf[leaf_nid]
        if leaf_nid in stage_leaves or value.depends_on_progress or bool(factor_states(value.factor)):
            absorbed.add(leaf_nid)
    absorbed.update(_final_copy_chain(ir, graph, evaluation.stages[-1].reducer_leaf))
    for leaf_nid in tuple(absorbed):
        output = graph.output_by_leaf[leaf_nid]
        absorbed.update(graph.initializers_by_tensor.get(output, ()))

    external_inputs: list[str] = []
    for leaf_nid in graph.leaves:
        if leaf_nid not in absorbed:
            continue
        for slot, tensor in graph.input_tensors_by_leaf[leaf_nid].items():
            producer = graph.predecessors[leaf_nid][slot]
            if producer not in absorbed and tensor not in external_inputs:
                external_inputs.append(tensor)

    external_outputs: list[str] = []
    for leaf_nid in graph.leaves:
        if leaf_nid not in absorbed:
            continue
        outside = [consumer for consumer in graph.successors[leaf_nid] if consumer not in absorbed]
        if outside:
            output = graph.output_by_leaf[leaf_nid]
            if output not in external_outputs:
                external_outputs.append(output)

    final_chain = _final_copy_chain(ir, graph, evaluation.stages[-1].reducer_leaf)
    final_leaf = final_chain[-1] if final_chain else evaluation.stages[-1].reducer_leaf
    final_output = graph.output_by_leaf[final_leaf]
    valid_boundary = external_outputs == [final_output]
    chunk_sizes = _chunk_sizes(ir, absorbed, progress_axis)
    result: OnlineFusionMatch | None = None
    if valid_boundary and chunk_sizes:
        order = {leaf: index for index, leaf in enumerate(graph.leaves)}
        derivation = tuple(sorted(absorbed, key=order.__getitem__))
        blocks = tuple(owning_block(ir.tree, leaf) for leaf in derivation)
        result = OnlineFusionMatch(
            progress_axis=progress_axis,
            progress_extent=ir.axis_extent(progress_axis),
            stages=evaluation.stages,
            derivation_leaves=derivation,
            absorbed_blocks=blocks,
            external_inputs=tuple(external_inputs),
            external_outputs=tuple(external_outputs),
            chunk_sizes=chunk_sizes,
            deferred_factor=_detect_deferred_factor(graph, evaluation),
        )
    return result


def _detect_deferred_factor(graph: ValueGraph, evaluation: AlgebraEvaluation) -> DeferredFactor | None:
    """Find a final broadcast state factor that can move after the recurrence."""
    result: DeferredFactor | None = None
    if evaluation.stages:
        stage_index = len(evaluation.stages) - 1
        final_stage = evaluation.stages[stage_index]
        split = _split_reciprocal_factor(final_stage.factor, stage_index)
        if split is not None:
            deferred, recurrence_factor = split
            assert isinstance(deferred, UnaryFactor)
            assert isinstance(deferred.operand, StateFactor)
            if _is_positive_sum_stage(graph, evaluation, deferred.operand.stage):
                producers = [
                    leaf
                    for leaf in graph.leaves
                    if evaluation.values_by_leaf[leaf].factor == deferred
                    and _is_reciprocal_producer(graph.contracts[leaf])
                ]
                candidates: list[DeferredFactor] = []
                final_ancestors = _ancestors(graph, {final_stage.reducer_leaf})
                for producer in producers:
                    candidates.extend(
                        _deferred_factor_candidates(
                            graph,
                            evaluation,
                            final_ancestors,
                            stage_index,
                            final_stage.factor,
                            deferred,
                            recurrence_factor,
                            producer,
                        )
                    )
                if len(candidates) == 1:
                    result = candidates[0]
        if result is None:
            result = _detect_fully_deferred_factor(graph, evaluation, stage_index)
    return result


def _detect_fully_deferred_factor(
    graph: ValueGraph, evaluation: AlgebraEvaluation, stage_index: int
) -> DeferredFactor | None:
    """Defer a uniquely produced broadcast factor in its entirety."""
    result: DeferredFactor | None = None
    final_stage = evaluation.stages[stage_index]
    deferred = final_stage.factor
    states = factor_states(deferred)
    if deferred is not None and states and all(index < stage_index for index in states):
        producers = [
            leaf
            for leaf in graph.leaves
            if evaluation.values_by_leaf[leaf].factor == deferred
            and not evaluation.values_by_leaf[leaf].depends_on_progress
            and _is_unary_pointwise_producer(graph.contracts[leaf])
        ]
        final_ancestors = _ancestors(graph, {final_stage.reducer_leaf})
        candidates: list[DeferredFactor] = []
        for producer in producers:
            candidates.extend(
                _deferred_factor_candidates(
                    graph, evaluation, final_ancestors, stage_index, final_stage.factor, deferred, None, producer
                )
            )
        if len(candidates) == 1:
            result = candidates[0]
    return result


def _split_reciprocal_factor(
    expression: FactorExpression | None, stage_index: int
) -> tuple[FactorExpression, FactorExpression | None] | None:
    """Separate one exact reciprocal of an earlier state from a product."""
    factors = _flatten_product(expression)
    reciprocal_indices = [
        index
        for index, factor in enumerate(factors)
        if isinstance(factor, UnaryFactor)
        and factor.operator == "reciprocal"
        and factor.scale == 1.0
        and factor.bias == 0.0
        and isinstance(factor.operand, StateFactor)
        and factor.operand.stage < stage_index
    ]
    result: tuple[FactorExpression, FactorExpression | None] | None = None
    if len(factors) > 1 and len(reciprocal_indices) == 1:
        reciprocal_index = reciprocal_indices[0]
        remaining = tuple(factor for index, factor in enumerate(factors) if index != reciprocal_index)
        result = factors[reciprocal_index], _multiply_factors(remaining)
    return result


def _flatten_product(expression: FactorExpression | None) -> tuple[FactorExpression, ...]:
    """Return the ordered leaves of a multiplication expression."""
    result: tuple[FactorExpression, ...] = ()
    if expression is not None:
        if isinstance(expression, BinaryFactor) and expression.operator == "multiply":
            result = (*_flatten_product(expression.left), *_flatten_product(expression.right))
        else:
            result = (expression,)
    return result


def _multiply_factors(factors: tuple[FactorExpression, ...]) -> FactorExpression | None:
    """Rebuild a multiplication expression after removing one factor."""
    result: FactorExpression | None = None
    for factor in factors:
        result = factor if result is None else BinaryFactor(operator="multiply", left=result, right=factor)
    return result


def _is_reciprocal_producer(contract: OperatorContract) -> bool:
    """Return whether a contract is an exact unary reciprocal."""
    return (
        isinstance(contract, PointwiseContract)
        and contract.operator == "reciprocal"
        and len(contract.input_operands) == 1
        and contract.scale == 1.0
        and contract.bias == 0.0
    )


def _is_unary_pointwise_producer(contract: OperatorContract) -> bool:
    """Return whether one contract materializes a unary pointwise factor."""
    return isinstance(contract, PointwiseContract) and len(contract.input_operands) == 1


def _is_positive_sum_stage(graph: ValueGraph, evaluation: AlgebraEvaluation, stage_index: int) -> bool:
    """Prove that one state is a nonempty additive reduction of exponentials."""
    result = False
    if 0 <= stage_index < len(evaluation.stages):
        stage = evaluation.stages[stage_index]
        reducer = graph.contracts[stage.reducer_leaf]
        if isinstance(reducer, ReductionContract) and stage.combinator.combiner == "add":
            producer = graph.predecessors[stage.reducer_leaf].get(reducer.input_operand)
            if producer is not None:
                producer_contract = graph.contracts[producer]
                result = (
                    isinstance(producer_contract, PointwiseContract)
                    and producer_contract.operator == "exp"
                    and len(producer_contract.input_operands) == 1
                )
    return result


def _deferred_factor_candidates(
    graph: ValueGraph,
    evaluation: AlgebraEvaluation,
    final_ancestors: set[int],
    stage_index: int,
    final_factor: FactorExpression | None,
    deferred: FactorExpression,
    recurrence_factor: FactorExpression | None,
    producer: int,
) -> list[DeferredFactor]:
    """Return shape-preserving multiply bypasses for one reciprocal producer."""
    candidates: list[DeferredFactor] = []
    successors = graph.successors[producer]
    if len(successors) == 1:
        combine = successors[0]
        contract = graph.contracts[combine]
        if (
            combine in final_ancestors
            and isinstance(contract, PointwiseContract)
            and contract.operator == "multiply"
            and len(contract.input_operands) == 2
            and evaluation.values_by_leaf[combine].factor == final_factor
            and _has_unique_transparent_path(graph, combine, evaluation.stages[stage_index].reducer_leaf)
        ):
            inputs = graph.input_tensors_by_leaf[combine]
            producer_slots = [
                slot
                for slot in contract.input_operands
                if graph.predecessors[combine].get(slot) == producer
                and evaluation.values_by_tensor[inputs[slot]].factor == deferred
            ]
            if len(producer_slots) == 1:
                factor_slot = producer_slots[0]
                passthrough_slots = [slot for slot in contract.input_operands if slot != factor_slot]
                passthrough = passthrough_slots[0]
                source = inputs[passthrough]
                output = graph.output_by_leaf[combine]
                source_factor = evaluation.values_by_tensor[source].factor
                if (
                    factor_slot in contract.broadcast_operands
                    and passthrough not in contract.broadcast_operands
                    and graph.tensor_axes[source] == graph.tensor_axes[output]
                    and source_factor == recurrence_factor
                ):
                    candidates.append(
                        DeferredFactor(
                            stage=stage_index,
                            factor=deferred,
                            recurrence_factor=recurrence_factor,
                            producer_leaf=producer,
                            bypass_leaf=combine,
                            passthrough_operand=passthrough,
                        )
                    )
    return candidates


def _has_unique_transparent_path(graph: ValueGraph, start: int, final_reducer: int) -> bool:
    """Check for one copy/permutation path from ``start`` to the final reducer."""
    current = start
    valid = True
    while valid and current != final_reducer:
        successors = graph.successors[current]
        valid = len(successors) == 1
        if valid:
            current = successors[0]
            if current != final_reducer:
                valid = isinstance(graph.contracts[current], (CopyContract, PermutationContract))
    return valid and current == final_reducer


def _ancestors(graph: ValueGraph, starts: set[int]) -> set[int]:
    """Return semantic producer ancestors including ``starts``."""
    result = set(starts)
    stack = list(starts)
    while stack:
        leaf = stack.pop()
        for producer in graph.predecessors[leaf].values():
            if producer is not None and producer not in result:
                result.add(producer)
                stack.append(producer)
    return result


def _final_copy_chain(ir: KernelIR, graph: ValueGraph, reducer_leaf: int) -> tuple[int, ...]:
    """Follow unique on-chip copies after the final reducer."""
    chain: list[int] = []
    current = reducer_leaf
    complete = False
    while not complete:
        consumers = graph.successors[current]
        if len(consumers) != 1:
            complete = True
        else:
            consumer = consumers[0]
            contract = graph.contracts[consumer]
            output = graph.output_by_leaf[consumer]
            if not isinstance(contract, CopyContract) or ir.buffer(output).location == "shared_hbm":
                complete = True
            else:
                chain.append(consumer)
                current = consumer
    return tuple(chain)


def _chunk_sizes(ir: KernelIR, absorbed: set[int], progress_axis: str) -> tuple[int, ...]:
    """Enumerate divisors tileable by every operation in the chain."""
    extent = ir.axis_extent(progress_axis)
    sizes: list[int] = []
    for size in range(1, extent + 1):
        valid = extent % size == 0
        for leaf_nid in absorbed:
            leaf = ir.tree.isa(leaf_nid)
            block = ir.tree.block(owning_block(ir.tree, leaf_nid))
            for abstract, concrete in block.axis_map.items():
                if concrete != progress_axis:
                    continue
                minimum = leaf.op_cls.MIN_TILE_SIZE.get(abstract, 1)
                maximum = leaf.op_cls.MAX_TILE_SIZE.get(abstract)
                tile = size if maximum is None else min(size, maximum)
                valid = valid and size >= minimum and size % tile == 0
        if valid:
            sizes.append(size)
    return tuple(sizes)


__all__ = [
    "BinaryFactor",
    "ConstantFactor",
    "FactorExpression",
    "OnlineFusionMatch",
    "OnlineFusionStage",
    "StateFactor",
    "UnaryFactor",
    "ValueGraph",
    "build_value_graph",
    "detect_complete_online_fusion",
    "detect_online_fusion",
]
