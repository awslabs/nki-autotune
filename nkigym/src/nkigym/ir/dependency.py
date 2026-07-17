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

from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx

from nkigym.ir.arith.expr import to_affine
from nkigym.ir.interval import regions_disjoint
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree

_HAZARD_PRIORITY: dict[str, int] = {"RAW": 3, "WAW": 2, "WAR": 1}


@dataclass(frozen=True)
class _BlockInfo:
    """Cached read/write regions, the buffers they touch, and enclosing-loop extents."""

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
        closure = nx.transitive_closure(self.graph, reflexive=False)
        self._closure: nx.DiGraph = nx.DiGraph()
        self._closure.add_nodes_from(closure.nodes(data=True))
        self._closure.add_edges_from(closure.edges(data=True))

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

    def first_backward_edge(self, moved_leaf_nid: int, tree: KernelTree | None = None) -> tuple[int, int] | None:
        """Return the first dependency edge incident to ``moved_leaf_nid`` that
        points backward in the execution order of ``tree``, else ``None``.

        One rule, no edge-kind. Each node has a preorder span ``[start, end]``
        over the tree (a leaf is a point; a loop spans its whole subtree). An
        edge ``a -> b`` ("a before b") is satisfied iff ``span(a).end <
        span(b).start`` and backward otherwise. A carry edge to a loop and a
        flow edge to a leaf are checked identically; the loop's wider span
        encodes "outside-and-before the whole loop".

        Edge *directions* always come from ``self.graph`` — this graph's
        producer->consumer orientation, frozen at construction. To test a
        *proposed* move, build this ``Dependency`` on the **original** program
        (correct directions) and pass the **moved** tree as ``tree`` so spans
        are read from the new positions. Rebuilding ``Dependency`` on the moved
        tree instead would be wrong: ``_build`` re-derives every flow edge from
        execution order, so a producer sunk past its consumer silently flips
        from RAW ``producer->consumer`` to WAR ``consumer->producer`` and the
        violation disappears. ``tree`` defaults to ``self._tree`` for the
        pure same-tree check.
        """
        eval_tree = tree if tree is not None else self._tree
        order = {n: i for i, n in enumerate(eval_tree.preorder())}

        def span(nid: int) -> tuple[float, float]:
            idxs = [order[d] for d in (eval_tree.descendants(nid) | {nid}) if d in order]
            if not idxs:
                raise KeyError(f"dependency endpoint {nid} absent from the evaluated tree")
            return (min(idxs), max(idxs))

        def enclosing_loops(nid: int) -> list[int]:
            return [a for a in eval_tree.ancestors(nid) if isinstance(eval_tree.data(a), ForNode)]

        return self._first_backward(moved_leaf_nid, span, eval_tree, enclosing_loops)

    def first_backward_edge_for_insertion(
        self, moved_leaf_nid: int, target_loop_nid: int, index: int
    ) -> tuple[int, int] | None:
        """Pure ordering check for splicing ``moved_leaf_nid`` under
        ``target_loop_nid`` at child slot ``index`` — no tree mutation.

        Equivalent to deep-copying, running ``_move``, rebuilding ``Dependency``
        on the moved tree and calling :meth:`first_backward_edge`, but O(edges)
        with no copy. Directions come from ``self.graph`` (build this
        ``Dependency`` on the original program); positions come from
        ``self._tree`` with the moved leaf relocated to its effective slot.
        ``index`` follows the ``_splice_under_target`` convention: ``-1``
        append, ``-2`` prepend, ``>=0`` explicit slot.

        The move relocates only the moved block's subtree; every dependency
        partner keeps its identity, so its position is read from the original
        tree with two adjustments that the physical splice would induce:

        - **Exclude the moved subtree** from each partner's span. A partner that
          enclosed the moved block before the move (or the target's children
          list, when re-moving an already-nested block) must not keep counting
          the relocated nodes at their old positions.
        - **Grow enclosing partners** to cover the new slot. Splicing the moved
          leaf under ``target_loop_nid`` makes it a descendant of the target and
          of every ancestor loop the target sits in, so a partner that
          **encloses the insertion point** has its span extended by the moved
          position. This is the carry-loop case ``K-loop -> drain``: sinking the
          drain *inside* the K loop must read as backward.

        Span-promotion evaluates the moved leaf's enclosing loops as the
        TARGET's nest (``target_loop_nid`` + its ForNode ancestors), since the
        splice makes it their descendant; every other endpoint keeps its own
        ``self._tree`` ForNode ancestors minus the moved subtree.
        """
        order: dict[int, float] = {n: i for i, n in enumerate(self._tree.preorder())}
        owner_block = self._owner_block.get(moved_leaf_nid, moved_leaf_nid)
        moved_subtree = self._tree.descendants(owner_block) | {owner_block}
        moved_pos = self._effective_insertion_position(order, target_loop_nid, index, moved_subtree)
        enclosers = set(self._tree.ancestors(target_loop_nid)) | {target_loop_nid}
        target_loops = [
            n
            for n in (target_loop_nid, *self._tree.ancestors(target_loop_nid))
            if isinstance(self._tree.data(n), ForNode)
        ]

        def span(nid: int) -> tuple[float, float]:
            if nid == moved_leaf_nid:
                return (moved_pos, moved_pos)
            positions = [order[d] for d in (self._tree.descendants(nid) | {nid}) - moved_subtree if d in order]
            if nid in enclosers:
                positions.append(moved_pos)
            if not positions:
                raise KeyError(f"dependency endpoint {nid} absent from the tree")
            return (min(positions), max(positions))

        def enclosing_loops(nid: int) -> list[int]:
            if nid == moved_leaf_nid:
                return target_loops
            return [
                a
                for a in self._tree.ancestors(nid)
                if a not in moved_subtree and isinstance(self._tree.data(a), ForNode)
            ]

        return self._first_backward(moved_leaf_nid, span, self._tree, enclosing_loops)

    def _effective_insertion_position(
        self, order: dict[int, float], target_loop_nid: int, index: int, moved_subtree: set[int]
    ) -> float:
        """Half-integer preorder position the moved leaf takes under the target.

        The leaf lands among ``target_loop_nid``'s children at the splice slot.
        Its position sits just after the node it follows: the target loop itself
        when prepending (before child 0), else the subtree-max of the preceding
        child. The ``+0.5`` keeps it strictly between adjacent integer indices so
        the span comparison orders it correctly against every other node.

        ``moved_subtree`` is excluded from the target's children first, matching
        ``_splice_under_target`` which detaches the moved block before indexing.
        Without this, re-moving a block that is already a child of the target
        (a prior compute_at nested it there) would count the block itself as a
        preceding sibling and place the leaf one slot too early.
        """
        children = [c for c in self._tree.children(target_loop_nid) if c not in moved_subtree]
        if index == -1:
            pos = len(children)
        elif index == -2:
            pos = 0
        elif index >= 0:
            pos = index
        else:
            raise ValueError(f"unsupported index {index} (use -1 append, -2 prepend, or >=0)")
        if pos <= 0 or not children:
            anchor = order[target_loop_nid]
        else:
            preceding = children[min(pos, len(children)) - 1]
            anchor = max(order[d] for d in (self._tree.descendants(preceding) | {preceding}))
        return anchor + 0.5

    def _first_backward(
        self,
        moved_leaf_nid: int,
        span: Callable[[int], tuple[float, float]],
        eval_tree: KernelTree,
        enclosing_loops: Callable[[int], list[int]],
    ) -> tuple[int, int] | None:
        """Return the first edge incident to ``moved_leaf_nid`` that ``span`` ranks
        backward after per-tensor span-promotion, else ``None``.

        Each edge carries the ``tensor`` its two leaves conflict on. Before the
        ``span(a).end < span(b).start`` test, each endpoint's span is promoted to
        any enclosing loop across which its access to that tensor is invariant and
        the tensor is carried (a live-across accumulator). ``enclosing_loops(nid)``
        returns the ForNode ancestor nids of an endpoint at its evaluated position
        — for the moved leaf under an insertion query, the TARGET's ancestors.
        """
        result: tuple[int, int] | None = None
        for a, b, attrs in self.graph.edges(data=True):
            if a != moved_leaf_nid and b != moved_leaf_nid:
                continue
            tensor = attrs.get("tensor")
            if tensor is None:
                continue
            span_a = _promoted_span(span(a), a, tensor, eval_tree, enclosing_loops(a), span)
            span_b = _promoted_span(span(b), b, tensor, eval_tree, enclosing_loops(b), span)
            if not (span_a[1] < span_b[0]):
                result = (a, b)
                break
        return result

    def chains(self) -> dict[str, list[int]]:
        """Return a copy of :attr:`touches_by_tensor` for safe iteration."""
        return {name: list(chain) for name, chain in self.touches_by_tensor.items()}

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
        """Insert a hazard edge, skipping self-loops and missing producers.

        The edge records both the hazard ``kind`` and the ``tensor`` the two
        leaves conflict on. Span-promotion reads ``tensor`` to decide, per edge,
        whether the shared buffer is carried across an enclosing loop.
        """
        result: None = None
        if producer is None or producer == consumer:
            result = None
        elif self._provably_disjoint(producer, consumer, tensor, kind):
            result = None
        else:
            keep = True
            if self.graph.has_edge(producer, consumer):
                current = self.graph.edges[producer, consumer]["kind"]
                if _HAZARD_PRIORITY[kind] <= _HAZARD_PRIORITY[current]:
                    keep = False
            if keep:
                self.graph.add_edge(producer, consumer, kind=kind, tensor=tensor)
        return result

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


def _leaf_operand_regions(tree: KernelTree, leaf_nid: int, tensor: str, rmw_only: bool) -> list[BufferRegion]:
    """Regions of ``tensor`` bound by ``leaf_nid``'s operands.

    With ``rmw_only`` True, only ``RMW_OPERANDS`` slots are considered — the
    read-modify-written accumulators (matmul ``dst``, tensor_tensor ``data1``)
    that can carry a value across a loop.
    """
    data = tree.data(leaf_nid)
    regions: list[BufferRegion] = []
    if isinstance(data, ISANode):
        slots = data.op_cls.RMW_OPERANDS if rmw_only else data.operand_bindings.keys()
        for slot in slots:
            region = data.operand_bindings.get(slot)
            if region is not None and region.tensor == tensor:
                regions.append(region)
    return regions


def _access_invariant_across(tree: KernelTree, leaf_nid: int, loop_var: str, tensor: str) -> bool:
    """True iff ``leaf_nid``'s access to ``tensor`` does NOT depend on ``loop_var``.

    Invariant means ``loop_var`` appears in no axis offset (``lo``) of any operand
    region naming ``tensor`` — so every iteration of the loop touches the same
    slice (a live-across / replicated access). A leaf that touches no such region
    is NOT invariant (there is no access to be invariant about).
    """
    regions = _leaf_operand_regions(tree, leaf_nid, tensor, rmw_only=False)
    invariant = bool(regions)
    for region in regions:
        if any(loop_var in to_affine(lo) for lo, _w in region.ranges):
            invariant = False
            break
    return invariant


def _tensor_carried_across(tree: KernelTree, loop_nid: int, tensor: str) -> bool:
    """True iff ``tensor`` is accumulated (live-carried) across ``loop_nid``.

    Two conditions. (1) Some ISA leaf inside the loop RMWs ``tensor`` invariantly
    across it — an accumulator whose slice is the same every iteration. (2) NO
    plain-write init of ``tensor`` is invariant across the loop and enclosed by
    it: such a write (e.g. a memset on a non-``RMW_OPERANDS`` slot) re-establishes
    the accumulator every iteration, so the loop RE-INITIALIZES rather than
    carries. Post-RFactor the matmul's psum rmw is invariant across BOTH ko and
    ki, but the per-ko memset sits inside ko and re-zeros it — so ko is a re-init
    loop (not carried) while ki (no enclosed init) is the true accumulation carry.
    A leaf that itself RMWs the tensor is its accumulator store-back (e.g. the
    fold's ``dst`` aliasing ``data1``), NOT a re-init — only a SEPARATE plain-write
    leaf (a memset) re-initializes. This is role-blind: it reads regions +
    ``RMW_OPERANDS`` only, never the axis role (RFactor flips ko/ki to PARALLEL
    yet psum still carries across ki).
    """
    loop = tree.data(loop_nid)
    assert isinstance(loop, ForNode), f"_tensor_carried_across: {loop_nid} is not a ForNode"
    loop_var = loop.loop_var
    has_invariant_rmw = False
    has_enclosed_init = False
    for nid in tree.descendants(loop_nid):
        data = tree.data(nid)
        if not isinstance(data, ISANode):
            continue
        rmw_regions = _leaf_operand_regions(tree, nid, tensor, rmw_only=True)
        if rmw_regions and not any(loop_var in to_affine(lo) for region in rmw_regions for lo, _w in region.ranges):
            has_invariant_rmw = True
        if rmw_regions:
            continue
        for slot, region in data.operand_bindings.items():
            if region.tensor != tensor:
                continue
            if slot in data.op_cls.RMW_OPERANDS:
                continue
            if slot in getattr(data.op_cls, "INPUT_OPERANDS", frozenset()):
                continue
            if not any(loop_var in to_affine(lo) for lo, _w in region.ranges):
                has_enclosed_init = True
    return has_invariant_rmw and not has_enclosed_init


def _promoted_span(
    base: tuple[float, float],
    endpoint_nid: int,
    tensor: str,
    eval_tree: KernelTree,
    enclosing_loops: list[int],
    span_of_loop: Callable[[int], tuple[float, float]],
) -> tuple[float, float]:
    """Widen ``base`` to any enclosing loop across which ``endpoint_nid``'s access
    to ``tensor`` is invariant and ``tensor`` is carried.

    A carried, loop-invariant access is live across the whole loop, so its
    effective span is the loop's span, not the leaf point. Promoting BOTH
    endpoints of a carried edge is what turns "memset lexically first inside K"
    into a backward edge (memset.end == K.end is not < matmul.start inside K).
    """
    lo, hi = base
    for loop_nid in enclosing_loops:
        loop = eval_tree.data(loop_nid)
        if not isinstance(loop, ForNode):
            continue
        if not _access_invariant_across(eval_tree, endpoint_nid, loop.loop_var, tensor):
            continue
        if not _tensor_carried_across(eval_tree, loop_nid, tensor):
            continue
        l_lo, l_hi = span_of_loop(loop_nid)
        lo = min(lo, l_lo)
        hi = max(hi, l_hi)
    return (lo, hi)


__all__ = ["Dependency"]
