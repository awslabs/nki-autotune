"""Producer-consumer dependency graph over :class:`ISANode` leaves.

The :class:`Dependency` class scans a :class:`KernelTree` in pre-order
DFS (emission order) and builds an ``nx.DiGraph`` whose nodes are the
integer ids of the tree's :class:`ISANode` leaves. An edge ``p -> c``
means ``p`` must execute before ``c`` to preserve program semantics;
the edge's ``kind`` attribute names the hazard (``"RAW"``, ``"WAW"``,
``"WAR"``).

Rewrite atoms such as ``ComputeAt`` use the graph to check that a
proposed move never places a consumer before its producer — the edge
set is the source of truth for legality. A per-tensor index
(:attr:`touches_by_tensor`) also provides the raw "who touched this
tensor and in what order" chain (e.g. ``sbuf_lhs_T -> [1, 10, 27]``)
that is useful for debugging and for reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from nkigym.ir._mermaid import ClassStyle, Flowchart, render_png
from nkigym.ir.tree import ISANode, KernelTree
from nkigym.ops.alloc import NKIAlloc

_DEPENDENCY_STYLES: list[ClassStyle] = [
    ClassStyle(name="alloc", fill="#fef", stroke="#963"),
    ClassStyle(name="leaf", fill="#efe", stroke="#363"),
]

_HAZARD_PRIORITY: dict[str, int] = {"RAW": 3, "WAW": 2, "WAR": 1}


@dataclass(frozen=True)
class _LeafInfo:
    """Cached reads/writes summary for a single leaf."""

    op_name: str
    is_alloc: bool
    reads: frozenset[str]
    writes: frozenset[str]
    rmw: frozenset[str]

    @property
    def read_set(self) -> frozenset[str]:
        """Tensor names this leaf reads (including RMW slots)."""
        return self.reads | self.rmw

    @property
    def write_set(self) -> frozenset[str]:
        """Tensor names this leaf writes (including RMW slots)."""
        return self.writes | self.rmw


class Dependency:
    """Producer-consumer graph over the leaves of a :class:`KernelTree`.

    Attributes:
        graph: Directed graph. Nodes are leaf ids; edges ``p -> c``
            carry ``kind`` ∈ {``"RAW"``, ``"WAW"``, ``"WAR"``}. When
            the same pair has multiple hazards, the first one recorded
            (pre-order walk; RAW > WAW > WAR) wins.
        touches_by_tensor: Tensor name → ordered list of leaf ids that
            read / write / rmw it, in emission order.
        leaves: Leaf ids in emission order.
    """

    def __init__(self, tree: KernelTree) -> None:
        """Scan ``tree`` in pre-order and build the dependency graph."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self.touches_by_tensor: dict[str, list[int]] = {}
        self.leaves: list[int] = []
        self._build(tree)
        self._closure: nx.DiGraph = nx.transitive_closure(self.graph, reflexive=False)

    def info(self, nid: int) -> "_LeafInfo":
        """Return the cached :class:`_LeafInfo` for leaf ``nid``."""
        return self.graph.nodes[nid]["info"]

    def direct_producers(self, nid: int) -> list[int]:
        """Return leaf ids that ``nid`` directly depends on."""
        return list(self.graph.predecessors(nid))

    def direct_consumers(self, nid: int) -> list[int]:
        """Return leaf ids that directly depend on ``nid``."""
        return list(self.graph.successors(nid))

    def producers(self, nid: int) -> set[int]:
        """Return every transitive producer of ``nid``."""
        return set(self._closure.predecessors(nid))

    def consumers(self, nid: int) -> set[int]:
        """Return every transitive consumer of ``nid``."""
        return set(self._closure.successors(nid))

    def must_precede(self, producer: int, consumer: int) -> bool:
        """Return True if ``producer`` must execute before ``consumer``."""
        return self._closure.has_edge(producer, consumer)

    def chains(self) -> dict[str, list[int]]:
        """Return a copy of :attr:`touches_by_tensor` for safe iteration."""
        return {name: list(chain) for name, chain in self.touches_by_tensor.items()}

    def dump(self, cache_dir: str | Path) -> None:
        """Write ``dependency.mmd`` and ``dependency.png`` into ``cache_dir``."""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        mmd_path = cache_path / "dependency.mmd"
        png_path = cache_path / "dependency.png"
        mmd_path.write_text(_to_mermaid(self), encoding="utf-8")
        render_png(mmd_path, png_path)

    def _build(self, tree: KernelTree) -> None:
        """Populate ``graph``, ``touches_by_tensor``, and ``leaves``."""
        last_writer: dict[str, int] = {}
        prior_readers: dict[str, list[int]] = {}
        for nid in tree.preorder():
            data = tree.data(nid)
            if not isinstance(data, ISANode):
                continue
            info = _LeafInfo(
                op_name=data.op_cls.__name__,
                is_alloc=data.op_cls is NKIAlloc,
                reads=frozenset(data.reads),
                writes=frozenset(data.writes),
                rmw=frozenset(data.rmw),
            )
            self.graph.add_node(nid, info=info)
            self.leaves.append(nid)
            for name in info.read_set | info.write_set:
                self.touches_by_tensor.setdefault(name, []).append(nid)
            self._record_hazards(nid, info, last_writer, prior_readers)
            for name in info.write_set:
                last_writer[name] = nid
                prior_readers.pop(name, None)
            for name in info.read_set - info.write_set:
                prior_readers.setdefault(name, []).append(nid)

    def _record_hazards(
        self, nid: int, info: _LeafInfo, last_writer: dict[str, int], prior_readers: dict[str, list[int]]
    ) -> None:
        """Add RAW/WAW/WAR edges from this leaf's current operands."""
        for name in info.read_set:
            self._try_edge(last_writer.get(name), nid, "RAW")
        for name in info.write_set:
            self._try_edge(last_writer.get(name), nid, "WAW")
            for prior_r in prior_readers.get(name, ()):
                self._try_edge(prior_r, nid, "WAR")

    def _try_edge(self, producer: int | None, consumer: int, kind: str) -> None:
        """Insert a hazard edge, skipping self-loops and missing producers."""
        if producer is None or producer == consumer:
            return
        if self.graph.has_edge(producer, consumer):
            current = self.graph.edges[producer, consumer]["kind"]
            if _HAZARD_PRIORITY[kind] <= _HAZARD_PRIORITY[current]:
                return
        self.graph.add_edge(producer, consumer, kind=kind)


def _to_mermaid(dep: Dependency) -> str:
    """Render ``dep`` to a Mermaid ``flowchart LR`` source string."""
    flow = Flowchart(direction="LR", styles=_DEPENDENCY_STYLES)
    for nid in dep.leaves:
        info = dep.info(nid)
        node_id = f"n{nid}"
        flow.add_node(node_id, f'{node_id}["{_leaf_label(nid, info)}"]', "alloc" if info.is_alloc else "leaf")
    for producer, consumer, attrs in dep.graph.edges(data=True):
        flow.add_edge(f"n{producer}", f"n{consumer}", label=attrs["kind"])
    return flow.render()


def _leaf_label(nid: int, info: _LeafInfo) -> str:
    """Build the Mermaid label for a dependency-graph leaf."""
    if info.is_alloc:
        return f"#{nid} alloc<br/>{min(info.writes)}"
    parts: list[str] = [f"#{nid} {info.op_name}"]
    if info.reads:
        parts.append(f"reads={','.join(sorted(info.reads))}")
    if info.writes:
        parts.append(f"writes={','.join(sorted(info.writes))}")
    if info.rmw:
        parts.append(f"rmw={','.join(sorted(info.rmw))}")
    return "<br/>".join(parts)


__all__ = ["Dependency"]
