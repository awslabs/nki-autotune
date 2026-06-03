"""Producer-consumer dependency graph over ISA leaves.

The :class:`Dependency` class scans a :class:`KernelTree` in pre-order
DFS and builds an ``nx.DiGraph`` whose nodes are ISA-leaf nids (each owned
by exactly one leaf :class:`BlockNode`). An edge ``p -> c`` means ``p``
must execute before ``c``. Public queries accept either a block nid
(legacy callers) or a leaf nid; ``_resolve`` maps block→leaf, and a leaf
nid maps to itself.

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
from nkigym.ir.expr import to_affine
from nkigym.ir.interval import regions_disjoint
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree, role_of
from nkigym.ops.base import AxisRole

_DEPENDENCY_STYLES: list[ClassStyle] = [ClassStyle(name="block", fill="#efe", stroke="#363")]

_HAZARD_PRIORITY: dict[str, int] = {"RAW": 3, "WAW": 2, "WAR": 1}


@dataclass(frozen=True)
class _BlockInfo:
    """Cached read/write regions, the buffers they touch, and enclosing-loop extents."""

    name: str
    reads: frozenset[str]
    writes: frozenset[str]
    read_regions: tuple[BufferRegion, ...]
    write_regions: tuple[BufferRegion, ...]
    extents: dict[str, int]
    buffers: dict[str, Buffer]


class Dependency:
    """Producer-consumer graph keyed on ISA-leaf nids (one per leaf :class:`BlockNode`)."""

    def __init__(self, tree: KernelTree) -> None:
        """Scan ``tree`` and build the leaf-keyed dependency graph."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self.touches_by_tensor: dict[str, list[int]] = {}
        self.blocks: list[int] = []
        self._leaf_of_block: dict[int, int] = {}
        self._owner_block: dict[int, int] = {}
        self._tree = tree
        self._build(tree)
        self._closure: nx.DiGraph = nx.transitive_closure(self.graph, reflexive=False)

    def _resolve(self, nid: int) -> int:
        """Map a block nid to its owned ISA-leaf nid; a leaf/loop nid maps to itself."""
        return self._leaf_of_block.get(nid, nid)

    def info(self, nid: int) -> _BlockInfo:
        """Return the cached :class:`_BlockInfo` for ``nid``."""
        return self.graph.nodes[self._resolve(nid)]["info"]

    def direct_producers(self, nid: int) -> list[int]:
        """Return leaf ids that ``nid`` directly depends on."""
        return list(self.graph.predecessors(self._resolve(nid)))

    def direct_consumers(self, nid: int) -> list[int]:
        """Return leaf ids that directly depend on ``nid``."""
        return list(self.graph.successors(self._resolve(nid)))

    def producers(self, nid: int) -> set[int]:
        """Return every transitive producer of ``nid``."""
        return set(self._closure.predecessors(self._resolve(nid)))

    def consumers(self, nid: int) -> set[int]:
        """Return every transitive consumer of ``nid``."""
        return set(self._closure.successors(self._resolve(nid)))

    def must_precede(self, producer: int, consumer: int) -> bool:
        """Return True if ``producer`` must execute before ``consumer``."""
        return self._closure.has_edge(self._resolve(producer), self._resolve(consumer))

    def first_backward_edge(self, moved_leaf_nid: int) -> tuple[int, int] | None:
        """Return the first dependency edge incident to ``moved_leaf_nid`` that
        points backward in this tree's execution order, else ``None``.

        One rule, no edge-kind. Each node has a preorder span ``[start, end]``
        over the tree (a leaf is a point; a loop spans its whole subtree). An
        edge ``a -> b`` ("a before b") is satisfied iff ``span(a).end <
        span(b).start`` and backward otherwise. A carry edge to a loop and a
        flow edge to a leaf are checked identically; the loop's wider span
        encodes "outside-and-before the whole loop". Callers that want to test
        a *proposed* move build the moved tree, construct a fresh
        ``Dependency`` on it, and call this with the moved leaf nid.
        """
        order = {n: i for i, n in enumerate(self._tree.preorder())}

        def span(nid: int) -> tuple[int, int]:
            idxs = [order[d] for d in (self._tree.descendants(nid) | {nid}) if d in order]
            return (min(idxs), max(idxs))

        result: tuple[int, int] | None = None
        for a, b in self.graph.edges():
            if a != moved_leaf_nid and b != moved_leaf_nid:
                continue
            if not (span(a)[1] < span(b)[0]):
                result = (a, b)
                break
        return result

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
        """Populate the graph by walking ISA leaves in execution order.

        A dependency node is an ISA leaf nid, keyed by the leaf rather than its
        owning :class:`BlockNode`. Each dependency block owns exactly one direct
        ISA leaf, so block and leaf form a bijection recorded in
        ``_leaf_of_block`` / ``_owner_block``. Co-location can nest one such
        block inside another (e.g. a sunk load block under the matmul's block);
        both still own exactly one leaf each. Leaves are processed in pre-order
        so the hazard walk sees writes and reads in the order the hardware
        executes them, not in tree pre-order (which lists an enclosing block
        before the producer block nested within it).
        """
        buffers = self._buffer_map(tree)
        last_writer: dict[str, int] = {}
        prior_readers: dict[str, list[int]] = {}
        for leaf_nid, block_nid in self._leaves_in_execution_order(tree):
            self._leaf_of_block[block_nid] = leaf_nid
            self._owner_block[leaf_nid] = block_nid
            block = tree.data(block_nid)
            assert isinstance(block, BlockNode)
            info = self._summarise(block_nid, block, tree, buffers)
            self.graph.add_node(leaf_nid, info=info)
            self.blocks.append(leaf_nid)
            for name in info.reads | info.writes:
                self.touches_by_tensor.setdefault(name, []).append(leaf_nid)
            self._record_hazards(leaf_nid, info, last_writer, prior_readers)
            for name in info.writes:
                last_writer[name] = leaf_nid
                prior_readers.pop(name, None)
            for name in info.reads - info.writes:
                prior_readers.setdefault(name, []).append(leaf_nid)
        self._add_carry_edges(tree)

    def _add_carry_edges(self, tree: KernelTree) -> None:
        """For each leaf with carry loops, add producer->loop and loop->consumer edges.

        A buffer carried across a non-PARALLEL loop ``L`` must be fully
        produced before ``L`` (init dominates) and only consumed after ``L``
        (drain post-dominates). Producers/consumers of the carried buffer are
        the graph's existing writers/readers of that tensor (``touches_by_tensor``
        filtered by write/read side). The reducer leaf itself is exempt.
        """
        for leaf_nid in list(self.graph.nodes):
            carries = _carry_loops_of_leaf(tree, leaf_nid)
            for loop_nid, tensor in carries.items():
                self.graph.add_node(loop_nid)
                for other in self.touches_by_tensor.get(tensor, ()):
                    if other == leaf_nid:
                        continue
                    info = self.graph.nodes[other]["info"]
                    if tensor in info.writes:
                        self.graph.add_edge(other, loop_nid, kind="CARRY")
                    if tensor in info.reads and tensor not in info.writes:
                        self.graph.add_edge(loop_nid, other, kind="CARRY")

    @staticmethod
    def _leaves_in_execution_order(tree: KernelTree) -> list[tuple[int, int]]:
        """Return (leaf_nid, owning_block_nid) pairs in ISA pre-order.

        Each ISA leaf is mapped to its nearest enclosing :class:`BlockNode`;
        walking leaves in pre-order yields the owning blocks in execution
        order. A block owning no ISA leaf (the synthetic root, or a pure
        loop-carrier) carries no hazard and never appears here.
        """
        ordered: list[tuple[int, int]] = []
        seen: set[int] = set()
        for leaf in tree.preorder():
            if not isinstance(tree.data(leaf), ISANode):
                continue
            owner = next(a for a in reversed(tree.ancestors(leaf)) if isinstance(tree.data(a), BlockNode))
            if owner in seen:
                raise AssertionError(f"block {owner} owns more than one ISA leaf; dependency model requires one")
            seen.add(owner)
            ordered.append((leaf, owner))
        return ordered

    @staticmethod
    def _buffer_map(tree: KernelTree) -> dict[str, Buffer]:
        """Collect every Buffer declared anywhere in the tree."""
        out: dict[str, Buffer] = {}
        for nid in tree.blocks():
            blk = tree.data(nid)
            assert isinstance(blk, BlockNode)
            for buf in blk.alloc_buffers:
                out[buf.name] = buf
        return out

    def _summarise(self, nid: int, block: BlockNode, tree: KernelTree, buffers: dict[str, Buffer]) -> _BlockInfo:
        """Build _BlockInfo with tensor-name sets, regions, extents, and buffers."""
        extents: dict[str, int] = {}
        for d in tree.descendants(nid):
            dd = tree.data(d)
            if isinstance(dd, ForNode):
                extents[dd.loop_var] = dd.extent
        reads = {r.tensor for r in block.reads}
        writes = {w.tensor for w in block.writes}
        return _BlockInfo(
            name=_block_name(block),
            reads=frozenset(reads),
            writes=frozenset(writes),
            read_regions=tuple(block.reads),
            write_regions=tuple(block.writes),
            extents=extents,
            buffers=buffers,
        )

    def _record_hazards(
        self, nid: int, info: _BlockInfo, last_writer: dict[str, int], prior_readers: dict[str, list[int]]
    ) -> None:
        for name in info.reads:
            self._try_edge(last_writer.get(name), nid, "RAW", name)
        for name in info.writes:
            self._try_edge(last_writer.get(name), nid, "WAW", name)
            for prior_r in prior_readers.get(name, ()):
                self._try_edge(prior_r, nid, "WAR", name)

    def _regions_for(self, nid: int, tensor: str, kind: str) -> tuple[BufferRegion, ...]:
        """Regions of ``tensor`` touched by block ``nid`` on the read or write side."""
        info = self.graph.nodes[nid]["info"]
        side = info.write_regions if kind == "write" else info.read_regions
        return tuple(r for r in side if r.tensor == tensor)

    def _try_edge(self, producer: int | None, consumer: int, kind: str, tensor: str) -> None:
        """Insert a hazard edge, skipping self-loops and missing producers."""
        if producer is None or producer == consumer:
            return
        if self._provably_disjoint(producer, consumer, tensor, kind):
            return
        if self.graph.has_edge(producer, consumer):
            current = self.graph.edges[producer, consumer]["kind"]
            if _HAZARD_PRIORITY[kind] <= _HAZARD_PRIORITY[current]:
                return
        self.graph.add_edge(producer, consumer, kind=kind)

    def _provably_disjoint(self, producer: int, consumer: int, tensor: str, kind: str) -> bool:
        """True iff every producer-region/consumer-region pair on ``tensor`` is disjoint.

        RAW: producer writes, consumer reads. WAW: both write. WAR: producer
        reads, consumer writes. If the tensor has no Buffer (a kernel param),
        treat as full-tensor → never disjoint (keep the edge).
        """
        pinfo = self.graph.nodes[producer]["info"]
        cinfo = self.graph.nodes[consumer]["info"]
        if tensor not in pinfo.buffers:
            return False
        buf = pinfo.buffers[tensor]
        prod_side = "write" if kind in ("RAW", "WAW") else "read"
        cons_side = "read" if kind == "RAW" else "write"
        prod_regions = self._regions_for(producer, tensor, prod_side)
        cons_regions = self._regions_for(consumer, tensor, cons_side)
        extents = {**pinfo.extents, **cinfo.extents}
        for pr in prod_regions:
            for cr in cons_regions:
                if not regions_disjoint(pr, cr, buf, buf, extents):
                    return False
        return True


def _enclosing_block_nid(tree: KernelTree, nid: int) -> int:
    """Return the nearest BlockNode ancestor of ``nid``."""
    result: int | None = None
    for anc in reversed(tree.ancestors(nid)):
        if isinstance(tree.data(anc), BlockNode):
            result = anc
            break
    if result is None:
        raise ValueError(f"no enclosing BlockNode for {nid}")
    return result


def _loopvar_to_axis(block: BlockNode) -> dict[str, str]:
    """Map each loop_var bound by the block to its concrete iter-var axis (via iter_values affine)."""
    out: dict[str, str] = {}
    for iv, value in zip(block.iter_vars, block.iter_values):
        for name in to_affine(value):
            if name is not None:
                out[name] = iv.axis
    return out


def _carry_loops_of_leaf(tree: KernelTree, leaf_nid: int) -> dict[int, str]:
    """Map each enclosing non-PARALLEL loop of ``leaf_nid`` to the buffer it carries.

    A loop carries state when its bound axis has SEQUENTIAL or ACCUMULATION
    role for the leaf's enclosing block. The carried buffer is the leaf
    operand whose ``OPERAND_AXES`` tuple omits that loop's axis (the value
    live across the loop). Loops whose axis is PARALLEL, or which carry no
    such operand, are skipped.
    """
    data = tree.data(leaf_nid)
    assert isinstance(data, ISANode)
    block_nid = _enclosing_block_nid(tree, leaf_nid)
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)
    lv_to_axis = _loopvar_to_axis(block)
    inverse_axis_map = {concrete: abstract for abstract, concrete in block.axis_map.items()}
    op_axes = data.op_cls.OPERAND_AXES
    out: dict[int, str] = {}
    for anc in tree.ancestors(leaf_nid):
        anc_data = tree.data(anc)
        if not isinstance(anc_data, ForNode):
            continue
        concrete = lv_to_axis.get(anc_data.loop_var)
        if concrete is None:
            continue
        if role_of(block, concrete) == AxisRole.PARALLEL:
            continue
        abstract = inverse_axis_map.get(concrete)
        for slot, axes in op_axes.items():
            if abstract is not None and abstract not in axes and slot in data.operand_bindings:
                if anc in out:
                    raise ValueError(
                        f"loop {anc} carries multiple operands ({out[anc]}, "
                        f"{data.operand_bindings[slot].tensor}); ambiguous carried buffer"
                    )
                out[anc] = data.operand_bindings[slot].tensor
    return out


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


__all__ = ["Dependency"]
