"""Make one reduction's supported first-write overwrite explicit."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.search.state_facts import operation_facts
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    software_pipeline_overlap_nodes,
)
from nkigym.transforms.eliminate_identity_initializer import (
    EliminateIdentityInitializer,
    EliminateIdentityInitializerOption,
    _InitializerMatch,
)
from nkigym.transforms.helper.canonical_rewrite import finalize_rewrite


@dataclass(frozen=True)
class SetFirstWriteOverwriteOption(TransformOption):
    """Identify one identity initializer and its one-step reduction."""

    initializer_block_nid: int
    reduction_block_nid: int
    tensor: str


class SetFirstWriteOverwrite(Transform[SetFirstWriteOverwriteOption]):
    """Mark one supported reduction invocation as a write-only first step."""

    def analyze(self, ir: KernelIR) -> list[SetFirstWriteOverwriteOption]:
        """Return reductions whose preceding identity makes overwrite equivalent."""
        facts = operation_facts(ir)
        if not facts.has_initializer or not facts.has_reduction:
            return []
        matcher = EliminateIdentityInitializer()
        options: list[SetFirstWriteOverwriteOption] = []
        overlap_nodes = software_pipeline_overlap_nodes(ir)
        for initializer_block_nid in ir.tree.blocks():
            candidate = matcher._candidate_option(ir, initializer_block_nid)
            if candidate is None:
                continue
            option = SetFirstWriteOverwriteOption(
                initializer_block_nid=candidate.initializer_block_nid,
                reduction_block_nid=candidate.reduction_block_nid,
                tensor=candidate.tensor,
            )
            if self._resolve(ir, option, overlap_nodes) is not None:
                options.append(option)
        return options

    def apply(self, ir: KernelIR, option: SetFirstWriteOverwriteOption) -> KernelIR:
        """Recheck, copy, and mark only the selected reduction invocation."""
        match = self._resolve(ir, option)
        if match is None:
            raise TransformLegalityError(f"illegal SetFirstWriteOverwrite option: {option}")
        result = copy_for_rewrite(ir)
        copied = self._resolve(result, option)
        if copied is None:
            raise AssertionError(f"SetFirstWriteOverwrite option disappeared after deepcopy: {option}")
        self._rewrite(result, copied)
        return result

    def _resolve(
        self, ir: KernelIR, option: SetFirstWriteOverwriteOption, overlap_nodes: frozenset[int] | None = None
    ) -> _InitializerMatch | None:
        """Resolve the shared identity/reduction proof before explicit marking."""
        candidate = EliminateIdentityInitializerOption(
            initializer_block_nid=option.initializer_block_nid,
            reduction_block_nid=option.reduction_block_nid,
            tensor=option.tensor,
        )
        return EliminateIdentityInitializer()._resolve(ir, candidate, explicit=False, overlap_nodes=overlap_nodes)

    def _rewrite(self, ir: KernelIR, match: _InitializerMatch) -> None:
        """Remove the destination read and set the first-write operation flag."""
        reduction = ir.tree.isa(match.reduction_leaf_nid)
        output = reduction.operand_bindings[match.output_operand]
        kwargs = reduction.op_cls.with_first_write_overwrite(match.output_operand, reduction.kwargs)
        ir.tree.graph.nodes[match.reduction_leaf_nid]["data"] = replace(reduction, kwargs=kwargs)
        block_nid = match.option.reduction_block_nid
        block = ir.tree.block(block_nid)
        reads = tuple(region for region in block.reads if region != output)
        if len(reads) + 1 != len(block.reads):
            raise AssertionError(f"reduction block {block_nid} does not read its destination exactly once")
        ir.tree.graph.nodes[block_nid]["data"] = replace(block, reads=reads)
        finalize_rewrite(ir)


__all__ = ["SetFirstWriteOverwrite", "SetFirstWriteOverwriteOption"]
