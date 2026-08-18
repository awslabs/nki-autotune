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
    """Identify one identity initializer and its reduction."""

    initializer_block_nid: int
    reduction_block_nid: int
    tensor: str
    initializer_leaf_nid: int | None = None


class SetFirstWriteOverwrite(Transform[SetFirstWriteOverwriteOption]):
    """Mark the first invocation of one supported reduction as write-only."""

    def analyze(self, ir: KernelIR) -> list[SetFirstWriteOverwriteOption]:
        """Return reductions whose preceding identity makes overwrite equivalent."""
        facts = operation_facts(ir)
        if not facts.has_initializer or not facts.has_reduction:
            return []
        matcher = EliminateIdentityInitializer()
        options: list[SetFirstWriteOverwriteOption] = []
        overlap_nodes = software_pipeline_overlap_nodes(ir)
        for initializer_leaf_nid in ir.tree.preorder():
            candidate = matcher._candidate_option(ir, initializer_leaf_nid)
            if candidate is None:
                continue
            option = SetFirstWriteOverwriteOption(
                initializer_block_nid=candidate.initializer_block_nid,
                reduction_block_nid=candidate.reduction_block_nid,
                tensor=candidate.tensor,
                initializer_leaf_nid=candidate.initializer_leaf_nid,
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
            initializer_leaf_nid=option.initializer_leaf_nid,
        )
        return EliminateIdentityInitializer()._resolve(ir, candidate, explicit=False, overlap_nodes=overlap_nodes)

    def _rewrite(self, ir: KernelIR, match: _InitializerMatch) -> None:
        """Mark one reduction axis for dynamic first-write lowering."""
        reduction = ir.tree.isa(match.reduction_leaf_nid)
        kwargs = reduction.op_cls.with_first_write_overwrite(
            match.output_operand, reduction.kwargs, match.reduction_axis
        )
        ir.tree.graph.nodes[match.reduction_leaf_nid]["data"] = replace(reduction, kwargs=kwargs)
        finalize_rewrite(ir)


__all__ = ["SetFirstWriteOverwrite", "SetFirstWriteOverwriteOption"]
