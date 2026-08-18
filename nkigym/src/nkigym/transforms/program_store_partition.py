"""Assign replicated post-reduction stores to disjoint logical NeuronCores."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import to_affine
from nkigym.ir.tree import BufferRegion, ISANode
from nkigym.ops.base import AxisRole, CopyContract, PointwiseContract
from nkigym.ops.sendrecv import NKISendRecv
from nkigym.ops.store import NKIStore
from nkigym.search.program_sharding import axis_loop_for_block, configured_program_shards, owning_block
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite


@dataclass(frozen=True)
class ProgramStorePartitionOption(TransformOption):
    """Assign one replicated store axis to ``programs`` logical cores."""

    target_nid: int
    axis: str
    programs: int


class ProgramStorePartition(Transform[ProgramStorePartitionOption]):
    """Partition one replicated post-reduction HBM store across programs."""

    def analyze(self, ir: KernelIR) -> list[ProgramStorePartitionOption]:
        """Return every legal output-axis ownership choice for a replicated store."""
        shards = configured_program_shards(ir)
        programs = next(iter(set(shards.values()))) if len(set(shards.values())) == 1 else None
        options: list[ProgramStorePartitionOption] = []
        if programs is not None:
            for leaf_nid in ir.dependency.graph.nodes:
                if _is_replicated_store(ir, leaf_nid):
                    block = ir.tree.block(owning_block(ir, leaf_nid))
                    axes = tuple(dict.fromkeys(block.axis_map[key] for key in ("F", "P") if key in block.axis_map))
                    for axis in axes:
                        option = ProgramStorePartitionOption(leaf_nid, axis, programs)
                        if _is_legal(ir, option):
                            options.append(option)
        return options

    def apply(self, ir: KernelIR, option: ProgramStorePartitionOption) -> KernelIR:
        """Re-check legality and assign one replicated store to disjoint programs."""
        _check_legality(ir, option)
        new_ir = copy_for_rewrite(ir)
        store = new_ir.tree.isa(option.target_nid)
        kwargs = {**store.kwargs, "program_ownership": (option.axis, option.programs)}
        new_ir.tree.graph.nodes[option.target_nid]["data"] = replace(store, kwargs=kwargs)
        return new_ir


def _is_legal(ir: KernelIR, option: ProgramStorePartitionOption) -> bool:
    """Return whether one store-ownership option is legal."""
    try:
        _check_legality(ir, option)
    except TransformLegalityError:
        return False
    return True


def _check_legality(ir: KernelIR, option: ProgramStorePartitionOption) -> None:
    """Require a replicated peer-reduced store indexed by a divisible parallel axis."""
    shards = configured_program_shards(ir)
    if option.programs not in shards.values():
        raise TransformLegalityError("ProgramStorePartition requires a matching program shard")
    if not _is_replicated_store(ir, option.target_nid):
        raise TransformLegalityError(f"ProgramStorePartition target {option.target_nid} is not a replicated store")
    store = ir.tree.isa(option.target_nid)
    if "program_ownership" in store.kwargs:
        raise TransformLegalityError("ProgramStorePartition target already has program ownership")
    block_nid = owning_block(ir, option.target_nid)
    if shards.keys() & set(ir.tree.ancestors(option.target_nid)):
        raise TransformLegalityError("ProgramStorePartition requires a store replicated on every program")
    block = ir.tree.block(block_nid)
    roles = {item.role for item in block.iter_vars if item.axis == option.axis}
    loop_nid = axis_loop_for_block(ir, block_nid, option.axis)
    if roles != {AxisRole.PARALLEL} or loop_nid is None:
        raise TransformLegalityError("ProgramStorePartition requires a materialized parallel output axis")
    loop = ir.tree.loop(loop_nid)
    destination = store.operand_bindings["dst"]
    if loop.extent % option.programs or not any(
        loop.loop_var in to_affine(lower) for lower, _width in destination.ranges
    ):
        raise TransformLegalityError("ProgramStorePartition requires a divisible axis-indexed HBM store")


def _is_replicated_store(ir: KernelIR, leaf_nid: int) -> bool:
    """Return whether one store consumes an identical peer-reduced value on every program."""
    shards = configured_program_shards(ir)
    if leaf_nid not in ir.tree.graph:
        return False
    store = ir.tree.data(leaf_nid)
    if not isinstance(store, ISANode) or store.op_cls is not NKIStore or set(shards.values()) != {2}:
        return False
    combined = _copy_origin(ir, leaf_nid, store.operand_bindings["src"])
    if combined is None:
        return False
    combine_nid, combined_region = combined
    combine = ir.tree.isa(combine_nid)
    contract = combine.op_cls.algebraic_contract(combine.kwargs)
    if (
        not isinstance(contract, PointwiseContract)
        or contract.operator not in {"add", "maximum", "multiply"}
        or len(contract.input_operands) != 2
        or combine.operand_bindings.get(contract.output_operand) != combined_region
    ):
        return False
    values = tuple(_program_value(ir, combine_nid, combine.operand_bindings[slot]) for slot in contract.input_operands)
    local = next((origin for peer, origin in values if not peer and origin is not None), None)
    remote = next((origin for peer, origin in values if peer and origin is not None), None)
    return local is not None and local == remote


def _program_value(
    ir: KernelIR, consumer_nid: int, region: BufferRegion
) -> tuple[bool, tuple[int, BufferRegion] | None]:
    """Return whether one value is peer-routed and its local value origin."""
    origin = _copy_origin(ir, consumer_nid, region)
    if origin is None:
        return False, None
    producer_nid, produced_region = origin
    producer = ir.tree.isa(producer_nid)
    if producer.op_cls is not NKISendRecv:
        return False, origin
    source = producer.operand_bindings.get("src")
    destination = producer.operand_bindings.get("dst")
    routed = (
        destination == produced_region
        and producer.kwargs.get("send_to_rank") == "program_peer"
        and producer.kwargs.get("recv_from_rank") == "program_peer"
    )
    return routed, _copy_origin(ir, producer_nid, source) if routed and source is not None else None


def _copy_origin(ir: KernelIR, consumer_nid: int, region: BufferRegion) -> tuple[int, BufferRegion] | None:
    """Follow value-preserving copies to one concrete producer and region."""
    current_nid = consumer_nid
    current_region = region
    while True:
        producers = [
            producer
            for producer in ir.dependency.direct_producers(current_nid)
            if current_region.tensor in ir.dependency.info(producer).writes
        ]
        if len(producers) != 1:
            return None
        producer_nid = producers[0]
        producer = ir.tree.isa(producer_nid)
        contract = producer.op_cls.algebraic_contract(producer.kwargs)
        if not isinstance(contract, CopyContract):
            written = tuple(
                region
                for region in ir.dependency.info(producer_nid).write_regions
                if region.tensor == current_region.tensor
            )
            return (producer_nid, written[0]) if len(written) == 1 else None
        if producer.operand_bindings.get(contract.output_operand) != current_region:
            return None
        source = producer.operand_bindings.get(contract.input_operand)
        if source is None:
            return None
        current_nid, current_region = producer_nid, source


__all__ = ["ProgramStorePartition", "ProgramStorePartitionOption"]
