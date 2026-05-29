"""Producer-consumer dependency graph over :class:`BlockNode` leaves.

The :class:`Dependency` class scans a :class:`KernelTree` in pre-order
DFS and builds an ``nx.DiGraph`` whose nodes are leaf-block nids
(blocks whose subtree contains exactly one ``ISANode``). An edge
``p -> c`` means ``p`` must execute before ``c``.

Edges are inserted whenever block ``b`` reads / writes a tensor that
some earlier block wrote / read with overlapping :class:`BufferRegion`
ranges. For canonical IR (every block under root, no compute_at), the
overlap test reduces to "same tensor"; transforms can produce nested
blocks where the per-iteration overlap matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from nkigym.ir._mermaid import ClassStyle, Flowchart, render_png
from nkigym.ir.tree import BlockNode, BufferRegion, KernelTree

_DEPENDENCY_STYLES: list[ClassStyle] = [ClassStyle(name="block", fill="#efe", stroke="#363")]

_HAZARD_PRIORITY: dict[str, int] = {"RAW": 3, "WAW": 2, "WAR": 1}


@dataclass(frozen=True)
class _BlockInfo:
    """Cached read / write tensor sets for a single leaf block."""

    name: str
    reads: frozenset[str]
    writes: frozenset[str]


class Dependency:
    """Producer-consumer graph over leaf :class:`BlockNode` nids."""

    def __init__(self, tree: KernelTree) -> None:
        """Scan ``tree`` and build the block-keyed dependency graph."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self.touches_by_tensor: dict[str, list[int]] = {}
        self.blocks: list[int] = []
        self._build(tree)
        self._closure: nx.DiGraph = nx.transitive_closure(self.graph, reflexive=False)

    def info(self, nid: int) -> _BlockInfo:
        """Return the cached :class:`_BlockInfo` for ``nid``."""
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
        """Populate the graph by walking leaf blocks (skipping the synthetic root)."""
        last_writer: dict[str, int] = {}
        prior_readers: dict[str, list[int]] = {}
        for nid in tree.blocks():
            block = tree.data(nid)
            assert isinstance(block, BlockNode)
            if not block.iter_vars and not block.reads and not block.writes:
                continue
            info = self._summarise(block)
            self.graph.add_node(nid, info=info)
            self.blocks.append(nid)
            for name in info.reads | info.writes:
                self.touches_by_tensor.setdefault(name, []).append(nid)
            self._record_hazards(nid, info, last_writer, prior_readers)
            for name in info.writes:
                last_writer[name] = nid
                prior_readers.pop(name, None)
            for name in info.reads - info.writes:
                prior_readers.setdefault(name, []).append(nid)

    def _summarise(self, block: BlockNode) -> _BlockInfo:
        """Collapse a block's BufferRegions to ``(reads, writes)`` tensor-name sets."""
        reads = {r.tensor for r in block.reads}
        writes = {w.tensor for w in block.writes}
        return _BlockInfo(name=_block_name(block), reads=frozenset(reads), writes=frozenset(writes))

    def _record_hazards(
        self, nid: int, info: _BlockInfo, last_writer: dict[str, int], prior_readers: dict[str, list[int]]
    ) -> None:
        for name in info.reads:
            self._try_edge(last_writer.get(name), nid, "RAW")
        for name in info.writes:
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


def _block_name(block: BlockNode) -> str:
    """Best-effort label for a block."""
    return block.annotations.get("name", "block")


def _to_mermaid(dep: Dependency) -> str:
    flow = Flowchart(direction="LR", styles=_DEPENDENCY_STYLES)
    for nid in dep.blocks:
        info = dep.info(nid)
        node_id = f"n{nid}"
        flow.add_node(node_id, f'{node_id}["{_label(nid, info)}"]', "block")
    for producer, consumer, attrs in dep.graph.edges(data=True):
        flow.add_edge(f"n{producer}", f"n{consumer}", label=attrs["kind"])
    return flow.render()


def _label(nid: int, info: _BlockInfo) -> str:
    parts: list[str] = [f"#{nid} {info.name}"]
    if info.reads:
        parts.append(f"reads={','.join(sorted(info.reads))}")
    if info.writes:
        parts.append(f"writes={','.join(sorted(info.writes))}")
    return "<br/>".join(parts)


def _bufferregion_overlaps(_a: BufferRegion, _b: BufferRegion) -> bool:
    """Stub for future per-region overlap analysis. Today's canonical IR doesn't need it; the
    block-pair tensor-name match is sufficient. Compute_at-driven nested blocks will exercise this."""
    return True


__all__ = ["Dependency"]
