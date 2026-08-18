"""Base classes for the rewrite-transform interface.

Each concrete transform under :mod:`nkigym.transforms` subclasses
:class:`Transform` and exposes:

* ``analyze(ir) -> list[TransformOption]`` — enumerate every legal
  option for this transform on ``ir``.
* ``apply(ir, option) -> KernelIR`` — re-check legality, deep-copy
  ``ir``, mutate the copy, return it. Raises
  :class:`TransformLegalityError` on illegal options. Loud failures
  only — no try/except recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar
from weakref import WeakKeyDictionary

import networkx as nx

from nkigym.ir import KernelIR, KernelTree
from nkigym.search.serialization import inherit_canonical_values

_PIPELINE_OVERLAPS: WeakKeyDictionary[KernelTree, frozenset[int]] = WeakKeyDictionary()


@dataclass(frozen=True)
class TransformOption:
    """Marker base for per-transform option payloads.

    Subclasses are frozen dataclasses (so options are hashable, useful
    for deduplication in samplers).
    """


class TransformLegalityError(ValueError):
    """Raised by :meth:`Transform.apply` when ``option`` is illegal for ``ir``."""


def _clone_tree(tree: KernelTree) -> KernelTree:
    """Clone mutable tree state while sharing frozen value payloads."""
    result = object.__new__(KernelTree)
    graph = nx.DiGraph()
    source = vars(tree.graph)
    target = vars(graph)
    target["graph"] = tree.graph.graph.copy()
    target["_node"] = {nid: attrs.copy() for nid, attrs in source["_node"].items()}
    target["_succ"] = {nid: edges.copy() for nid, edges in source["_succ"].items()}
    target["_pred"] = {nid: edges.copy() for nid, edges in source["_pred"].items()}
    target["_adj"] = target["_succ"]
    result.graph = graph
    result._next_id = tree.next_node_id
    result.root = tree.root
    return result


def copy_for_rewrite(ir: KernelIR) -> KernelIR:
    """Copy mutable IR state without cloning dependency data that will be rebuilt."""
    tree = _clone_tree(ir.tree)
    inherit_canonical_values(ir.tree, tree)
    if (overlap := _PIPELINE_OVERLAPS.get(ir.tree)) is not None:
        _PIPELINE_OVERLAPS[tree] = overlap
    return KernelIR(
        func_name=ir.func_name,
        param_names=list(ir.param_names),
        return_names=ir.return_names,
        tree=tree,
        dependency=ir.dependency,
        param_buffers=dict(ir.param_buffers),
    )


def software_pipeline_overlap_nodes(ir: KernelIR) -> frozenset[int]:
    """Return every node comparable with an active pipeline loop."""
    cached = _PIPELINE_OVERLAPS.get(ir.tree)
    if cached is not None:
        return cached
    result: set[int] = set()
    for block_nid in ir.tree.blocks():
        annotation = ir.tree.block(block_nid).annotations.get("software_pipeline")
        if annotation is None:
            continue
        loop_nid = annotation["loop_nid"]
        result.update((loop_nid, *ir.tree.ancestors(loop_nid), *ir.tree.descendants(loop_nid)))
    overlap = frozenset(result)
    _PIPELINE_OVERLAPS[ir.tree] = overlap
    return overlap


def invalidate_software_pipeline_overlap(tree: KernelTree) -> None:
    """Discard cached overlap facts after pipeline annotations change."""
    _PIPELINE_OVERLAPS.pop(tree, None)


def intersects_software_pipeline(
    ir: KernelIR, nids: tuple[int, ...], overlap_nodes: frozenset[int] | None = None
) -> bool:
    """Return whether any selected node overlaps an active pipeline scope."""
    active = software_pipeline_overlap_nodes(ir) if overlap_nodes is None else overlap_nodes
    return not active.isdisjoint(nids)


_OptionT = TypeVar("_OptionT", bound=TransformOption)


class Transform(Generic[_OptionT]):
    """Base class for stateless rewrite transforms.

    Subclasses override :meth:`analyze` and :meth:`apply`. Instances
    carry no state — the same instance can be reused across many
    ``ir``'s.
    """

    SPLIT_PREPARATION_DEPTH = 0

    def split_preparation_applicable(self, ir: KernelIR) -> bool:
        """Return whether legal splits may expose this transform."""
        _ = ir
        return self.SPLIT_PREPARATION_DEPTH > 0

    def split_preparation_ready(self, ir: KernelIR) -> bool:
        """Return whether the current state completes a useful split path."""
        return bool(self.analyze(ir))

    def analyze(self, ir: KernelIR) -> list[_OptionT]:
        """Return every legal option for this transform on ``ir``."""
        raise NotImplementedError

    def apply(self, ir: KernelIR, option: _OptionT) -> KernelIR:
        """Re-check legality, deep-copy ``ir``, mutate the copy, return it."""
        raise NotImplementedError


__all__ = [
    "Transform",
    "TransformLegalityError",
    "TransformOption",
    "copy_for_rewrite",
    "invalidate_software_pipeline_overlap",
    "intersects_software_pipeline",
    "software_pipeline_overlap_nodes",
]
