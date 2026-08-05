"""Contract-driven online-fusion transform."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.transforms._canonical_rewrite import finalize_rewrite, remove_buffers
from nkigym.transforms._online_fusion_analysis import detect_complete_online_fusion, detect_online_fusion
from nkigym.transforms._online_fusion_lowering import (
    can_lower_online_fusion,
    can_lower_online_fusion_prefix,
    lower_online_fusion,
    lower_online_fusion_prefix,
)
from nkigym.transforms._online_fusion_types import OnlineFusionMatch
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption

_INCREMENTAL_ANNOTATION = "online_fusion_incremental"


@dataclass(frozen=True)
class _IncrementalState:
    """Metadata needed to select and complete one retained recurrence prefix."""

    prefix_match_id: tuple[str, tuple[int, ...]]
    complete_match_id: tuple[str, tuple[int, ...]]
    chunk_size: int
    roots: tuple[int, ...]
    nodes: frozenset[int]
    buffers: tuple[str, ...]
    next_node_id: int


@dataclass(frozen=True)
class OnlineFusionOption(TransformOption):
    """One proven chain and selected sequential chunk size."""

    match_id: tuple[str, tuple[int, ...]]
    chunk_size: int


class OnlineFusion(Transform[OnlineFusionOption]):
    """Rewrite an algebraically separable reduction chain into online form."""

    def analyze(self, ir: KernelIR) -> list[OnlineFusionOption]:
        """Enumerate contract-proven and lowering-supported options."""
        options: list[OnlineFusionOption] = []
        state = _incremental_state(ir)
        if state is not None:
            if _incremental_prefix_intact(ir, state):
                probe = copy.deepcopy(ir)
                _remove_incremental_prefix(probe)
                matches = {match.match_id: match for match in detect_complete_online_fusion(probe)}
                match = matches.get(state.complete_match_id)
                chunk_size = state.chunk_size
                if match is not None and can_lower_online_fusion(probe, match, chunk_size):
                    options.append(OnlineFusionOption(match_id=match.match_id, chunk_size=chunk_size))
        else:
            for match in detect_online_fusion(ir):
                for chunk_size in match.chunk_sizes:
                    if match.incremental_prefix:
                        legal = can_lower_online_fusion_prefix(ir, match, chunk_size)
                    else:
                        legal = can_lower_online_fusion(ir, match, chunk_size)
                    if legal:
                        options.append(OnlineFusionOption(match_id=match.match_id, chunk_size=chunk_size))
        return options

    def apply(self, ir: KernelIR, option: OnlineFusionOption) -> KernelIR:
        """Re-check ``option``, deep-copy, and lower the recurrence."""
        state = _incremental_state(ir)
        if state is not None:
            result = self._apply_completion(ir, option, state)
        else:
            result = self._apply_initial(ir, option)
        return result

    def _apply_initial(self, ir: KernelIR, option: OnlineFusionOption) -> KernelIR:
        """Apply either a complete two-stage chain or the first recurrence prefix."""
        matches = {match.match_id: match for match in detect_online_fusion(ir)}
        match = matches.get(option.match_id)
        legal = match is not None and (
            can_lower_online_fusion_prefix(ir, match, option.chunk_size)
            if match.incremental_prefix
            else can_lower_online_fusion(ir, match, option.chunk_size)
        )
        if not legal or match is None:
            raise TransformLegalityError(f"illegal OnlineFusion option: {option}")

        result = copy.deepcopy(ir)
        copied_matches = {candidate.match_id: candidate for candidate in detect_online_fusion(result)}
        copied_match = copied_matches.get(option.match_id)
        if copied_match is None:
            raise AssertionError(f"OnlineFusion match {option.match_id} disappeared after deepcopy")
        if copied_match.incremental_prefix:
            complete = _matching_complete(result, copied_match)
            next_node_id = result.tree.next_node_id
            roots, buffers = lower_online_fusion_prefix(result, copied_match, option.chunk_size)
            root = result.tree.block(result.tree.root)
            annotations = dict(root.annotations)
            annotations[_INCREMENTAL_ANNOTATION] = _IncrementalState(
                prefix_match_id=copied_match.match_id,
                complete_match_id=complete.match_id,
                chunk_size=option.chunk_size,
                roots=roots,
                nodes=_nodes_under_roots(result, roots),
                buffers=buffers,
                next_node_id=next_node_id,
            )
            result.tree.graph.nodes[result.tree.root]["data"] = replace(root, annotations=annotations)
        else:
            lower_online_fusion(result, copied_match, option.chunk_size)
        return result

    def _apply_completion(self, ir: KernelIR, option: OnlineFusionOption, state: _IncrementalState) -> KernelIR:
        """Remove the retained prefix and lower its complete chain extension."""
        if (
            option.match_id != state.complete_match_id
            or option.chunk_size != state.chunk_size
            or not _incremental_prefix_intact(ir, state)
        ):
            raise TransformLegalityError(f"illegal OnlineFusion completion option: {option}")
        result = copy.deepcopy(ir)
        _remove_incremental_prefix(result)
        matches = {match.match_id: match for match in detect_complete_online_fusion(result)}
        match = matches.get(option.match_id)
        if match is None or not can_lower_online_fusion(result, match, option.chunk_size):
            raise TransformLegalityError(f"illegal OnlineFusion completion option: {option}")
        lower_online_fusion(result, match, option.chunk_size)
        return result


def _matching_complete(ir: KernelIR, prefix: OnlineFusionMatch) -> OnlineFusionMatch:
    """Return the unique complete chain beginning with ``prefix``."""
    candidates = [
        match
        for match in detect_complete_online_fusion(ir)
        if match.progress_axis == prefix.progress_axis
        and match.stages[: len(prefix.stages)] == prefix.stages
        and set(prefix.chunk_sizes) & set(match.chunk_sizes)
    ]
    if len(candidates) != 1:
        raise TransformLegalityError(
            f"online-fusion prefix {prefix.match_id} has {len(candidates)} complete extensions"
        )
    return candidates[0]


def _remove_incremental_prefix(ir: KernelIR) -> None:
    """Remove retained prefix blocks and restore the canonical node-id allocator."""
    state = _incremental_state(ir)
    if state is None:
        raise ValueError("IR has no incremental online-fusion prefix")
    if not _incremental_prefix_intact(ir, state):
        raise ValueError("incremental online-fusion prefix has been structurally modified")
    remove_buffers(ir, set(state.buffers))
    for block_nid in state.roots:
        ir.tree.graph.remove_nodes_from({block_nid, *ir.tree.descendants(block_nid)})
    annotations = dict(ir.tree.block(ir.tree.root).annotations)
    del annotations[_INCREMENTAL_ANNOTATION]
    restored_root = replace(ir.tree.block(ir.tree.root), annotations=annotations)
    ir.tree.graph.nodes[ir.tree.root]["data"] = restored_root
    if all(nid < state.next_node_id for nid in ir.tree.graph.nodes):
        ir.tree.restore_next_id(state.next_node_id)
    finalize_rewrite(ir)


def _nodes_under_roots(ir: KernelIR, roots: tuple[int, ...]) -> frozenset[int]:
    """Return each retained root and every node currently below it."""
    nodes: set[int] = set()
    for block_nid in roots:
        nodes.update({block_nid, *ir.tree.descendants(block_nid)})
    return frozenset(nodes)


def _incremental_prefix_intact(ir: KernelIR, state: _IncrementalState) -> bool:
    """Return whether the retained prefix can still be removed atomically."""
    intact = all(block_nid in ir.tree.graph and ir.tree.parent(block_nid) == ir.tree.root for block_nid in state.roots)
    if intact:
        intact = _nodes_under_roots(ir, state.roots) == state.nodes
    if intact:
        intact = set(state.buffers).issubset(ir.all_buffers())
    return intact


def _incremental_state(ir: KernelIR) -> _IncrementalState | None:
    """Return validated incremental metadata from the root."""
    value = ir.tree.block(ir.tree.root).annotations.get(_INCREMENTAL_ANNOTATION)
    if value is not None and not isinstance(value, _IncrementalState):
        raise ValueError(f"malformed {_INCREMENTAL_ANNOTATION} annotation: {value!r}")
    return value


__all__ = ["OnlineFusion", "OnlineFusionOption"]
