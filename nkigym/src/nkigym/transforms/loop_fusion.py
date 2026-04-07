"""Loop Fusion transform: merge adjacent fusion groups.

Two adjacent groups can fuse when: (1) the consumer reads the
producer's output, and (2) all dimensions of the intermediate
tensor are non-blocking for both producer and consumer.

Fusion sets ``buffer_degrees[intermediate] = 1`` (degree-1 buffer).
"""

from dataclasses import replace
from typing import ClassVar

from nkigym.codegen.render import KernelIR
from nkigym.transforms import Transform


class LoopFusion(Transform):
    """Fuse pairs of adjacent fusion groups."""

    NAME: ClassVar[str] = "loop_fusion"

    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        """Return all single-step fusion candidates.

        For each pair of adjacent fusion groups, checks fusion
        legality and returns a new KernelIR with those groups merged.
        """
        results: list[KernelIR] = []
        for i in range(len(ir.fusion_groups) - 1):
            group_a = ir.fusion_groups[i]
            group_b = ir.fusion_groups[i + 1]
            producer_idx = group_a[-1]
            consumer_idx = group_b[0]
            intermediate = self._find_intermediate(ir, producer_idx, consumer_idx)
            if intermediate is None:
                continue
            if not self._all_dims_nonblocking(ir, producer_idx, consumer_idx, intermediate):
                continue
            new_ir = self._fuse(ir, i, intermediate)
            results.append(new_ir)
        return results

    def _find_intermediate(self, ir: KernelIR, producer_idx: int, consumer_idx: int) -> str | None:
        """Return intermediate tensor name if consumer reads producer's output."""
        _, _, producer_output = ir.ops[producer_idx]
        _, consumer_operand_map, _ = ir.ops[consumer_idx]
        result = producer_output if producer_output in consumer_operand_map.values() else None
        return result

    def _all_dims_nonblocking(self, ir: KernelIR, producer_idx: int, consumer_idx: int, intermediate: str) -> bool:
        """Check that all intermediate dims are non-blocking for both ops."""
        tensor = ir.ctx.tensors[intermediate]

        producer_op = ir.ops[producer_idx][0]
        consumer_op = ir.ops[consumer_idx][0]
        producer_reverse = {v: k for k, v in ir.per_op_maps[producer_idx].items()}
        consumer_reverse = {v: k for k, v in ir.per_op_maps[consumer_idx].items()}

        ok = bool(tensor.dim_ids)
        for dim_id in tensor.dim_ids:
            producer_abstract = producer_reverse.get(dim_id)
            if producer_abstract and producer_abstract in producer_op.BLOCKING_AXES:
                ok = False
            consumer_abstract = consumer_reverse.get(dim_id)
            if consumer_abstract and consumer_abstract in consumer_op.BLOCKING_AXES:
                ok = False
        return ok

    def _fuse(self, ir: KernelIR, group_idx: int, intermediate: str) -> KernelIR:
        """Create a new KernelIR with groups at group_idx and group_idx+1 merged."""
        new_groups: list[list[int]] = []
        new_dim_orders: list[tuple[str, ...]] = []
        for i, group in enumerate(ir.fusion_groups):
            if i == group_idx:
                new_groups.append(group + ir.fusion_groups[i + 1])
                new_dim_orders.append(ir.ctx.tensors[intermediate].dim_ids)
            elif i == group_idx + 1:
                continue
            else:
                new_groups.append(group)
                new_dim_orders.append(ir.group_dim_orders[i])

        new_degrees = dict(ir.buffer_degrees)
        new_degrees[intermediate] = 1

        return replace(ir, fusion_groups=new_groups, buffer_degrees=new_degrees, group_dim_orders=new_dim_orders)
