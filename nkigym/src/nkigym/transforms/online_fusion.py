"""Contract-driven online-fusion transform."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, Buffer, ForNode, ISANode
from nkigym.transforms._canonical_rewrite import finalize_rewrite, remove_buffers
from nkigym.transforms._online_fusion_analysis import (
    build_value_graph,
    detect_complete_online_fusion,
    detect_online_fusion,
)
from nkigym.transforms._online_fusion_lowering import (
    OnlineFusionPrefixLowering,
    can_lower_online_fusion,
    can_lower_online_fusion_prefix,
    complete_online_fusion_prefix,
    lower_online_fusion,
    lower_online_fusion_prefix,
)
from nkigym.transforms._online_fusion_types import OnlineFusionMatch, ValueGraph
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption

_INCREMENTAL_ANNOTATION = "online_fusion_incremental"


@dataclass(frozen=True)
class _NodeSnapshot:
    """One removed tree node and its immutable payload."""

    nid: int
    payload: BlockNode | ForNode | ISANode


@dataclass(frozen=True)
class _RemovedBlock:
    """One reducer subtree removed by a live recurrence prefix."""

    root: int
    nodes: tuple[_NodeSnapshot, ...]
    edges: tuple[tuple[int, int], ...]
    previous_root: int | None
    next_root: int | None


@dataclass(frozen=True)
class _PrefixSnapshot:
    """Detached payload, topology, and buffer state for one live prefix."""

    nodes: tuple[_NodeSnapshot, ...]
    children: tuple[tuple[int, tuple[int, ...]], ...]
    buffers: tuple[Buffer, ...]


@dataclass(frozen=True)
class _IncrementalState:
    """Metadata needed to select and complete one live recurrence prefix."""

    prefix_match_id: tuple[str, tuple[int, ...]]
    complete_match_id: tuple[str, tuple[int, ...]]
    chunk_size: int
    snapshot: _PrefixSnapshot
    lowering: OnlineFusionPrefixLowering
    removed_blocks: tuple[_RemovedBlock, ...]


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
                completion = _resolve_completion(ir, state)
                if completion is not None:
                    _probe, match, _graph = completion
                    options.append(OnlineFusionOption(match_id=match.match_id, chunk_size=state.chunk_size))
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
            removed_blocks = _capture_removed_blocks(result, copied_match)
            lowering = lower_online_fusion_prefix(result, copied_match, complete, option.chunk_size)
            root = result.tree.block(result.tree.root)
            annotations = dict(root.annotations)
            annotations[_INCREMENTAL_ANNOTATION] = _IncrementalState(
                prefix_match_id=copied_match.match_id,
                complete_match_id=complete.match_id,
                chunk_size=option.chunk_size,
                snapshot=_capture_prefix(result, lowering.roots),
                lowering=lowering,
                removed_blocks=removed_blocks,
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
        completion = _resolve_completion(ir, state)
        if completion is None:
            raise TransformLegalityError(f"illegal OnlineFusion completion option: {option}")
        _probe, match, graph = completion
        result = copy.deepcopy(ir)
        copied_state = _incremental_state(result)
        if copied_state is None:
            raise AssertionError("OnlineFusion incremental state disappeared after deepcopy")
        complete_online_fusion_prefix(result, match, graph, copied_state.lowering, option.chunk_size)
        annotations = dict(result.tree.block(result.tree.root).annotations)
        del annotations[_INCREMENTAL_ANNOTATION]
        root = replace(result.tree.block(result.tree.root), annotations=annotations)
        result.tree.graph.nodes[result.tree.root]["data"] = root
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


def _capture_removed_blocks(ir: KernelIR, match: OnlineFusionMatch) -> tuple[_RemovedBlock, ...]:
    """Capture reducer subtrees so analysis can reconstruct the materialized graph."""
    siblings = ir.tree.children(ir.tree.root)
    records: list[_RemovedBlock] = []
    for stage in match.stages:
        root = stage.reducer_block
        index = siblings.index(root)
        node_ids = (root, *ir.tree.descendants(root))
        node_set = set(node_ids)
        records.append(
            _RemovedBlock(
                root=root,
                nodes=tuple(_NodeSnapshot(nid=nid, payload=ir.tree.data(nid)) for nid in node_ids),
                edges=tuple(
                    (parent, child) for parent, child in ir.tree.graph.edges if parent in node_set and child in node_set
                ),
                previous_root=siblings[index - 1] if index > 0 else None,
                next_root=siblings[index + 1] if index + 1 < len(siblings) else None,
            )
        )
    return tuple(records)


def _resolve_completion(
    ir: KernelIR, state: _IncrementalState
) -> tuple[KernelIR, OnlineFusionMatch, ValueGraph] | None:
    """Reconstruct and validate the current materialized completion path."""
    probe = copy.deepcopy(ir)
    _restore_materialized_probe(probe, state)
    matches = {match.match_id: match for match in detect_complete_online_fusion(probe)}
    match = matches.get(state.complete_match_id)
    result: tuple[KernelIR, OnlineFusionMatch, ValueGraph] | None = None
    if match is not None and can_lower_online_fusion(probe, match, state.chunk_size):
        result = (probe, match, build_value_graph(probe))
    return result


def _restore_materialized_probe(ir: KernelIR, state: _IncrementalState) -> None:
    """Remove the live prefix and restore its two original reducer blocks."""
    if not _incremental_prefix_intact(ir, state):
        raise ValueError("incremental online-fusion prefix has been structurally modified")
    added_buffers = set(state.lowering.added_buffers)
    if added_buffers:
        remove_buffers(ir, added_buffers)
    for block_nid in state.lowering.roots:
        ir.tree.graph.remove_nodes_from({block_nid, *ir.tree.descendants(block_nid)})
    for record in state.removed_blocks:
        if record.root in ir.tree.graph:
            ir.tree.graph.remove_nodes_from({record.root, *ir.tree.descendants(record.root)})
        for snapshot in record.nodes:
            if snapshot.nid in ir.tree.graph:
                raise ValueError(f"cannot restore occupied online-fusion node {snapshot.nid}")
            ir.tree.graph.add_node(snapshot.nid, data=snapshot.payload)
        ir.tree.graph.add_edges_from(record.edges)
        _attach_restored_root(ir, record)
    annotations = dict(ir.tree.block(ir.tree.root).annotations)
    del annotations[_INCREMENTAL_ANNOTATION]
    root = replace(ir.tree.block(ir.tree.root), annotations=annotations)
    ir.tree.graph.nodes[ir.tree.root]["data"] = root
    finalize_rewrite(ir)


def _attach_restored_root(ir: KernelIR, record: _RemovedBlock) -> None:
    """Attach one reducer at its original dataflow position."""
    siblings = ir.tree.children(ir.tree.root)
    if record.next_root in siblings:
        index = siblings.index(record.next_root)
    elif record.previous_root in siblings:
        index = siblings.index(record.previous_root) + 1
    else:
        index = len(siblings)
    order = siblings[:index] + [record.root] + siblings[index:]
    for child in siblings:
        ir.tree.graph.remove_edge(ir.tree.root, child)
    for child in order:
        ir.tree.graph.add_edge(ir.tree.root, child)


def _nodes_under_roots(ir: KernelIR, roots: tuple[int, ...]) -> frozenset[int]:
    """Return each retained root and every node currently below it."""
    nodes: set[int] = set()
    for block_nid in roots:
        nodes.update({block_nid, *ir.tree.descendants(block_nid)})
    return frozenset(nodes)


def _capture_prefix(ir: KernelIR, roots: tuple[int, ...]) -> _PrefixSnapshot:
    """Capture prefix state without aliasing mutable node payload dictionaries."""
    node_ids = _nodes_under_roots(ir, roots)
    snapshots = tuple(_NodeSnapshot(nid=nid, payload=copy.deepcopy(ir.tree.data(nid))) for nid in sorted(node_ids))
    children = tuple((nid, tuple(ir.tree.children(nid))) for nid in sorted(node_ids))
    tensor_names: set[str] = set()
    for nid in node_ids:
        node = ir.tree.data(nid)
        if isinstance(node, BlockNode):
            tensor_names.update(region.tensor for region in (*node.reads, *node.writes))
            tensor_names.update(buffer.name for buffer in node.alloc_buffers)
        elif isinstance(node, ISANode):
            tensor_names.update(region.tensor for region in node.operand_bindings.values())
    buffers = ir.all_buffers()
    buffer_snapshots = tuple(copy.deepcopy(buffers[name]) for name in sorted(tensor_names))
    return _PrefixSnapshot(nodes=snapshots, children=children, buffers=buffer_snapshots)


def _incremental_prefix_intact(ir: KernelIR, state: _IncrementalState) -> bool:
    """Return whether the retained prefix can still be removed atomically."""
    roots = state.lowering.roots
    intact = all(block_nid in ir.tree.graph and ir.tree.parent(block_nid) == ir.tree.root for block_nid in roots)
    if intact:
        intact = tuple(child for child in ir.tree.children(ir.tree.root) if child in roots) == roots
    node_snapshots = {snapshot.nid: snapshot.payload for snapshot in state.snapshot.nodes}
    if intact:
        intact = _nodes_under_roots(ir, roots) == frozenset(node_snapshots)
    if intact:
        intact = all(ir.tree.data(nid) == payload for nid, payload in node_snapshots.items())
    if intact:
        intact = all(tuple(ir.tree.children(nid)) == children for nid, children in state.snapshot.children)
    if intact:
        intact = set(state.lowering.added_buffers).issubset(ir.all_buffers())
    if intact:
        buffers = ir.all_buffers()
        intact = all(buffers.get(buffer.name) == buffer for buffer in state.snapshot.buffers)
    if intact:
        intact = (
            state.lowering.loop_nid in ir.tree.graph
            and state.lowering.carrier_nid in ir.tree.graph
            and all(block in ir.tree.graph for block in state.lowering.roll_forward_blocks)
        )
    return intact


def _incremental_state(ir: KernelIR) -> _IncrementalState | None:
    """Return validated incremental metadata from the root."""
    value = ir.tree.block(ir.tree.root).annotations.get(_INCREMENTAL_ANNOTATION)
    if value is not None and not isinstance(value, _IncrementalState):
        raise ValueError(f"malformed {_INCREMENTAL_ANNOTATION} annotation: {value!r}")
    return value


__all__ = ["OnlineFusion", "OnlineFusionOption"]
