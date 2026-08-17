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
from weakref import WeakKeyDictionary

import networkx as nx

from nkigym.ir.arith.expr import to_affine
from nkigym.ir.graph_index import DAGReachability, ordered_tree_topology
from nkigym.ir.interval import regions_disjoint
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree

_HAZARD_PRIORITY: dict[str, int] = {"RAW": 3, "WAW": 2, "WAR": 1}
_LEAF_OPERAND_REGIONS: WeakKeyDictionary[KernelTree, dict[tuple[int, str, bool], tuple[BufferRegion, ...]]] = (
    WeakKeyDictionary()
)
_ACCESS_INVARIANTS: WeakKeyDictionary[KernelTree, dict[tuple[int, str, str], bool]] = WeakKeyDictionary()
_CARRIED_TENSORS: WeakKeyDictionary[KernelTree, dict[tuple[int, str], bool]] = WeakKeyDictionary()
_Topology = tuple[dict[int, int], dict[int, tuple[int, ...]], dict[int, frozenset[int]]]


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
        self._order, self._ancestors, self._descendants = ordered_tree_topology(tree.graph, tree.root)
        self._topology_valid = True
        self._reachability = DAGReachability(self.graph)

    def _reachable(self, nid: int, backward: bool) -> frozenset[int]:
        """Return cached transitive predecessors or successors of one leaf."""
        return self._reachability.nodes(nid, backward)

    def _resolve(self, nid: int) -> int:
        """Map a block nid to its owned ISA-leaf nid; a leaf/loop nid maps to itself."""
        return self._leaf_of_block.get(nid, nid)

    def info(self, nid: int) -> _BlockInfo:
        """Return the cached :class:`_BlockInfo` for ``nid``."""
        return getattr(self.graph, "_node")[self._resolve(nid)]["info"]

    def direct_producers(self, nid: int) -> list[int]:
        """Return leaf ids that ``nid`` directly depends on."""
        return list(getattr(self.graph, "_pred")[self._resolve(nid)])

    def direct_consumers(self, nid: int) -> list[int]:
        """Return leaf ids that directly depend on ``nid``."""
        return list(getattr(self.graph, "_succ")[self._resolve(nid)])

    def producers(self, nid: int) -> set[int]:
        """Return every transitive producer of ``nid``."""
        return set(self._reachable(self._resolve(nid), True))

    def consumers(self, nid: int) -> set[int]:
        """Return every transitive consumer of ``nid``."""
        return set(self._reachable(self._resolve(nid), False))

    def must_precede(self, producer: int, consumer: int) -> bool:
        """Return True if ``producer`` must execute before ``consumer``."""
        return self._reachability.precedes(self._resolve(producer), self._resolve(consumer))

    def first_backward_edge_for_insertion(
        self, moved_leaf_nid: int, target_loop_nid: int, index: int, topology: _Topology | None = None
    ) -> tuple[int, int] | None:
        """Pure ordering check for splicing ``moved_leaf_nid`` under
        ``target_loop_nid`` at child slot ``index`` — no tree mutation.

        Computes the proposed execution spans in O(edges) without copying or
        mutating the tree. Directions come from ``self.graph`` (build this
        ``Dependency`` on the original program); positions come from ``self._tree``
        with the moved leaf relocated to its effective slot.
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

        ``topology`` may provide a snapshot already validated by a read-only
        analysis pass, avoiding repeated graph mutation checks for each slot.
        """
        order, ancestors, descendants = self._topology() if topology is None else topology
        owner = self._owner_block.get(moved_leaf_nid, moved_leaf_nid)
        moved_descendants = descendants[owner]
        children = [
            child for child in self._tree.children(target_loop_nid) if child != owner and child not in moved_descendants
        ]
        if index == -1:
            position = len(children)
        elif index == -2:
            position = 0
        elif index >= 0:
            position = index
        else:
            raise ValueError(f"unsupported index {index} (use -1 append, -2 prepend, or >=0)")
        if position <= 0 or not children:
            anchor = order[target_loop_nid]
        else:
            preceding = children[min(position, len(children)) - 1]
            anchor = order[preceding] + len(descendants[preceding])
        moved_position = anchor + 0.5
        target_loops = tuple(
            nid for nid in (target_loop_nid, *ancestors[target_loop_nid]) if isinstance(self._tree.data(nid), ForNode)
        )
        moved_spans: dict[str, tuple[float, float]] = {}
        static_spans = _PROMOTED_SPANS.setdefault(self, {})

        def promoted_moved(tensor: str) -> tuple[float, float]:
            """Return the moved leaf's span under the proposed target loops."""
            result = moved_spans.get(tensor)
            if result is None:
                lo = hi = moved_position
                for loop_nid in target_loops:
                    loop = self._tree.loop(loop_nid)
                    if _access_invariant_across(
                        self._tree, moved_leaf_nid, loop.loop_var, tensor
                    ) and _tensor_carried_across(self._tree, loop_nid, tensor):
                        loop_position = float(order[loop_nid])
                        lo = min(lo, loop_position)
                        hi = max(hi, loop_position)
                result = (lo, hi)
                moved_spans[tensor] = result
            return result

        def promoted_static(nid: int, tensor: str) -> tuple[float, float]:
            """Return one unchanged endpoint's cached carried-loop span."""
            key = (nid, tensor)
            result = static_spans.get(key)
            if result is None:
                lo = hi = float(order[nid])
                for loop_nid in ancestors[nid]:
                    loop = self._tree.data(loop_nid)
                    if not isinstance(loop, ForNode):
                        continue
                    if _access_invariant_across(self._tree, nid, loop.loop_var, tensor) and _tensor_carried_across(
                        self._tree, loop_nid, tensor
                    ):
                        loop_position = float(order[loop_nid])
                        lo = min(lo, loop_position)
                        hi = max(hi, loop_position)
                result = (lo, hi)
                static_spans[key] = result
            return result

        bounds = _INSERTION_BOUNDS.setdefault(self, {}).get(moved_leaf_nid)
        if bounds is None:
            incoming: dict[str, tuple[float, int]] = {}
            outgoing: dict[str, tuple[float, int]] = {}
            for first, attrs in getattr(self.graph, "_pred")[moved_leaf_nid].items():
                tensor = attrs.get("tensor")
                if isinstance(tensor, str):
                    high = promoted_static(first, tensor)[1]
                    if tensor not in incoming or high > incoming[tensor][0]:
                        incoming[tensor] = (high, first)
            for second, attrs in getattr(self.graph, "_succ")[moved_leaf_nid].items():
                tensor = attrs.get("tensor")
                if isinstance(tensor, str):
                    low = promoted_static(second, tensor)[0]
                    if tensor not in outgoing or low < outgoing[tensor][0]:
                        outgoing[tensor] = (low, second)
            bounds = (
                tuple((tensor, high, first) for tensor, (high, first) in incoming.items()),
                tuple((tensor, low, second) for tensor, (low, second) in outgoing.items()),
            )
            _INSERTION_BOUNDS.setdefault(self, {})[moved_leaf_nid] = bounds
        result: tuple[int, int] | None = None
        for tensor, high, first in bounds[0]:
            if high >= promoted_moved(tensor)[0]:
                result = (first, moved_leaf_nid)
                break
        if result is None:
            for tensor, low, second in bounds[1]:
                if promoted_moved(tensor)[1] >= low:
                    result = (moved_leaf_nid, second)
                    break
        return result

    def _topology(self) -> _Topology:
        """Return cached topology or rebuild it when callers mutate the tree."""
        if not self._topology_valid:
            self._order, self._ancestors, self._descendants = ordered_tree_topology(self._tree.graph, self._tree.root)
            self._topology_valid = True
        return self._order, self._ancestors, self._descendants

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


def _rmw_operand_slots(node: ISANode) -> frozenset[str]:
    """Return statically RMW slots plus input/output slots that alias exactly."""
    configured_rmw = node.op_cls.rmw_operands(node.kwargs)
    slots = set(configured_rmw)
    inputs = node.op_cls.INPUT_OPERANDS
    outputs = set(node.operand_bindings) - inputs - configured_rmw
    for input_slot in inputs:
        input_region = node.operand_bindings.get(input_slot)
        if input_region is None:
            continue
        for output_slot in outputs:
            if node.operand_bindings[output_slot] == input_region:
                slots.update((input_slot, output_slot))
    return frozenset(slots)


def _leaf_operand_regions(tree: KernelTree, leaf_nid: int, tensor: str, rmw_only: bool) -> tuple[BufferRegion, ...]:
    """Regions of ``tensor`` bound by ``leaf_nid``'s operands.

    With ``rmw_only`` True, only statically RMW slots or explicitly aliased
    input/output slots are considered. The latter covers an RFactor
    ``tensor_tensor(data1=acc, dst=acc)`` without declaring every SSA
    ``tensor_tensor`` operation read-modify-write.
    """
    cache = _LEAF_OPERAND_REGIONS.setdefault(tree, {})
    key = (leaf_nid, tensor, rmw_only)
    regions = cache.get(key)
    if regions is None:
        data = tree.data(leaf_nid)
        selected: list[BufferRegion] = []
        if isinstance(data, ISANode):
            slots = _rmw_operand_slots(data) if rmw_only else data.operand_bindings.keys()
            for slot in slots:
                region = data.operand_bindings.get(slot)
                if region is not None and region.tensor == tensor:
                    selected.append(region)
        regions = tuple(selected)
        cache[key] = regions
    return regions


def _access_invariant_across(tree: KernelTree, leaf_nid: int, loop_var: str, tensor: str) -> bool:
    """True iff ``leaf_nid``'s access to ``tensor`` does NOT depend on ``loop_var``.

    Invariant means ``loop_var`` appears in no axis offset (``lo``) of any operand
    region naming ``tensor`` — so every iteration of the loop touches the same
    slice (a live-across / replicated access). A leaf that touches no such region
    is NOT invariant (there is no access to be invariant about).
    """
    cache = _ACCESS_INVARIANTS.setdefault(tree, {})
    key = (leaf_nid, loop_var, tensor)
    invariant = cache.get(key)
    if invariant is None:
        regions = _leaf_operand_regions(tree, leaf_nid, tensor, rmw_only=False)
        invariant = bool(regions)
        for region in regions:
            if any(loop_var in to_affine(lo) for lo, _w in region.ranges):
                invariant = False
                break
        cache[key] = invariant
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
    static or explicitly aliased RMW operands only, never the axis role
    (RFactor flips ko/ki to PARALLEL yet psum still carries across ki).
    """
    cache = _CARRIED_TENSORS.setdefault(tree, {})
    key = (loop_nid, tensor)
    carried = cache.get(key)
    if carried is None:
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
            rmw_slots = _rmw_operand_slots(data)
            for slot, region in data.operand_bindings.items():
                if region.tensor != tensor:
                    continue
                if slot in rmw_slots:
                    continue
                if slot in getattr(data.op_cls, "INPUT_OPERANDS", frozenset()):
                    continue
                if not any(loop_var in to_affine(lo) for lo, _w in region.ranges):
                    has_enclosed_init = True
        carried = has_invariant_rmw and not has_enclosed_init
        cache[key] = carried
    return carried


_PROMOTED_SPANS: WeakKeyDictionary[Dependency, dict[tuple[int, str], tuple[float, float]]] = WeakKeyDictionary()
_INSERTION_BOUNDS: WeakKeyDictionary[
    Dependency, dict[int, tuple[tuple[tuple[str, float, int], ...], tuple[tuple[str, float, int], ...]]]
] = WeakKeyDictionary()


__all__ = ["Dependency"]
