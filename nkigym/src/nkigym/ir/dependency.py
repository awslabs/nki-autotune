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

from nkigym.ir.arith.expr import expr_variables
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
        self._order, self._ancestors, self._descendants = ordered_tree_topology(tree.graph, tree.root)
        self._topology_valid = True
        self._build(tree)
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
        self,
        moved_leaf_nid: int,
        target_loop_nid: int,
        index: int,
        topology: _Topology | None = None,
        moved_root_nid: int | None = None,
    ) -> tuple[int, int] | None:
        """Return the first frozen dependency violated by a proposed insertion.

        Place the moved leaf between original preorder positions, excluding its
        old child slot when moving among siblings. ``index`` follows the splice
        convention: -1 appends, -2 prepends, and nonnegative values select a slot.
        Targets must lie outside the moved subtree.

        A tensor carried across an enclosing loop expands the access span back
        to that loop. Ancestors precede their leaves, so this changes only the
        lower endpoint; the upper endpoint remains the instruction position.
        Evaluate the moved leaf under the target's loop nest and keep every
        other endpoint under its original nest. Incoming producers must end
        before the moved span starts, and outgoing consumers must start after
        it ends. Edge directions always come from the original dependency DAG.

        ``topology`` may supply the current analysis pass's preorder snapshot.
        Sibling indices and unchanged dependency bounds are cached on this
        sidecar; the query does not copy or mutate the program tree.
        """
        order, ancestors, descendants = self._topology() if topology is None else topology
        owner = self._owner_block.get(moved_leaf_nid, moved_leaf_nid) if moved_root_nid is None else moved_root_nid
        if target_loop_nid == owner or target_loop_nid in descendants[owner]:
            raise ValueError("insertion target must be outside the moved subtree")
        children, indices = self.child_order(target_loop_nid)
        removed = indices.get(owner)
        count = len(children) - (removed is not None)
        if index < -2:
            raise ValueError(f"unsupported index {index} (use -1 append, -2 prepend, or >=0)")
        position = count if index == -1 else 0 if index == -2 else index
        anchor = order[target_loop_nid]
        if position > 0 and count:
            slot = min(position, count) - 1
            if removed is not None and slot >= removed:
                slot += 1
            preceding = children[slot]
            anchor = order[preceding] + len(descendants[preceding])
        moved_position = anchor + 0.5
        loop_scopes = _LOOP_SCOPES.get(self)
        if loop_scopes is None:
            loop_scopes = {
                nid: tuple(
                    (parent, node.loop_var)
                    for parent in (*parents, nid)
                    if isinstance(node := self._tree.data(parent), ForNode)
                )
                for nid, parents in ancestors.items()
            }
            _LOOP_SCOPES[self] = loop_scopes
        static_spans = _PROMOTED_SPANS.setdefault(self, {})

        def lower_span(nid: int, tensor: str, start: float, loops: tuple[tuple[int, str], ...]) -> float:
            """Expand only the lower endpoint: enclosing loops precede the leaf."""
            for loop_nid, loop_var in loops:
                if _access_invariant_across(self._tree, nid, loop_var, tensor) and _tensor_carried_across(
                    self, loop_nid, tensor
                ):
                    start = min(start, float(order[loop_nid]))
            return start

        bounds = _INSERTION_BOUNDS.setdefault(self, {}).get(moved_leaf_nid)
        if bounds is None:
            incoming: dict[str, tuple[float, int]] = {}
            outgoing: dict[str, tuple[float, int]] = {}
            for first, attrs in getattr(self.graph, "_pred")[moved_leaf_nid].items():
                tensor = attrs.get("tensor")
                if isinstance(tensor, str):
                    high = float(order[first])
                    if tensor not in incoming or high > incoming[tensor][0]:
                        incoming[tensor] = (high, first)
            for second, attrs in getattr(self.graph, "_succ")[moved_leaf_nid].items():
                tensor = attrs.get("tensor")
                if isinstance(tensor, str):
                    key = (second, tensor)
                    if key not in static_spans:
                        static_spans[key] = lower_span(second, tensor, float(order[second]), loop_scopes[second])
                    low = static_spans[key]
                    if tensor not in outgoing or low < outgoing[tensor][0]:
                        outgoing[tensor] = (low, second)
            bounds = (
                tuple((tensor, high, first) for tensor, (high, first) in incoming.items()),
                tuple((tensor, low, second) for tensor, (low, second) in outgoing.items()),
            )
            _INSERTION_BOUNDS.setdefault(self, {})[moved_leaf_nid] = bounds
        result: tuple[int, int] | None = None
        for tensor, high, first in bounds[0]:
            if high >= lower_span(moved_leaf_nid, tensor, moved_position, loop_scopes[target_loop_nid]):
                result = (first, moved_leaf_nid)
                break
        if result is None:
            for tensor, low, second in bounds[1]:
                if moved_position >= low:
                    result = (moved_leaf_nid, second)
                    break
        return result

    def child_order(self, nid: int) -> tuple[tuple[int, ...], dict[int, int]]:
        """Return cached ordered children and their sibling indices."""
        cache = _CHILD_ORDER.setdefault(self, {})
        result = cache.get(nid)
        if result is None:
            children = tuple(self._tree.children(nid))
            result = children, {child: index for index, child in enumerate(children)}
            cache[nid] = result
        return result

    def _topology(self) -> _Topology:
        """Return cached topology or rebuild it when callers mutate the tree."""
        if not self._topology_valid:
            self._order, self._ancestors, self._descendants = ordered_tree_topology(self._tree.graph, self._tree.root)
            _CHILD_ORDER.pop(self, None)
            _LOOP_SCOPES.pop(self, None)
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
        buffers = self._buffers = self._buffer_map(tree)
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

    def _leaves_in_execution_order(self, tree: KernelTree) -> list[tuple[int, int]]:
        """Return (leaf_nid, owning_block_nid) pairs in ISA pre-order.

        Each ISA leaf is mapped to its nearest enclosing :class:`BlockNode`;
        walking leaves in pre-order yields the owning blocks in execution
        order. A block owning no ISA leaf (the synthetic root, or a pure
        loop-carrier) carries no hazard and never appears here.
        """
        ordered = [
            (leaf, next(a for a in reversed(self._ancestors[leaf]) if isinstance(tree.data(a), BlockNode)))
            for leaf in self._order
            if isinstance(tree.data(leaf), ISANode)
        ]
        if len({owner for _leaf, owner in ordered}) != len(ordered):
            raise AssertionError("dependency blocks must each own exactly one ISA leaf")
        return ordered

    @staticmethod
    def _buffer_map(tree: KernelTree) -> dict[str, Buffer]:
        """Collect every Buffer declared anywhere in the tree."""
        declarations = [buffer for nid in tree.blocks() for buffer in tree.block(nid).alloc_buffers]
        buffers = {buffer.name: buffer for buffer in declarations}
        if len(buffers) != len(declarations):
            raise ValueError("a buffer is declared by two blocks")
        return buffers

    def _summarise(self, nid: int, block: BlockNode, tree: KernelTree, buffers: dict[str, Buffer]) -> _BlockInfo:
        """Build _BlockInfo with tensor-name sets, regions, extents, and buffers."""
        _order, ancestors, descendants = self._topology()
        extents = {
            node.loop_var: node.extent for child in descendants[nid] if isinstance(node := tree.data(child), ForNode)
        }
        read_regions = tuple(block.reads) + tuple(
            region
            for ancestor in ancestors[nid]
            if isinstance(scope := tree.data(ancestor), BlockNode) and "predicate" in scope.annotations
            for region in scope.reads
        )
        reads = {r.tensor for r in read_regions}
        writes = {w.tensor for w in block.writes}
        return _BlockInfo(
            reads=frozenset(reads),
            writes=frozenset(writes),
            read_regions=read_regions,
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
            if any(loop_var in expr_variables(lo) for lo, _w in region.ranges):
                invariant = False
                break
        cache[key] = invariant
    return invariant


def _tensor_carried_across(dependency: Dependency, loop_nid: int, tensor: str) -> bool:
    """True iff ``tensor`` is accumulated (live-carried) across ``loop_nid``.

    Two conditions. (1) Some ISA leaf inside the loop RMWs ``tensor`` invariantly
    across it — an accumulator whose slice is the same every iteration. (2) NO
    plain-write init of ``tensor`` is invariant across the loop and enclosed by
    it: such a write (e.g. a memset on a non-``RMW_OPERANDS`` slot) re-establishes
    the accumulator every iteration, so the loop RE-INITIALIZES rather than
    carries. Post-RFactor the matmul's psum rmw is invariant across BOTH ko and
    ki, but the per-ko memset sits inside ko and re-zeros it — so ko is a re-init
    loop (not carried) while ki (no enclosed init) is the true accumulation carry.
    A leaf that itself RMWs the tensor is normally its accumulator store-back,
    not a re-init. An explicit dynamic first-write overwrite is the exception:
    it re-initializes enclosing loops whose coordinates do not index any leaf
    operand, while the indexed reduction loop remains carried. This is
    role-blind: it reads regions and configured RMW operands, never axis roles.
    The dependency index limits inspection to leaves that touch this tensor.
    """
    cache = _CARRIED_TENSORS.setdefault(tree := dependency._tree, {})
    key = (loop_nid, tensor)
    carried = cache.get(key)
    if carried is None:
        assert isinstance(loop := tree.data(loop_nid), ForNode), f"_tensor_carried_across: {loop_nid} is not a ForNode"
        has_invariant_rmw = has_enclosed_init = False
        for nid in dependency.touches_by_tensor.get(tensor, ()):
            if loop_nid not in dependency._topology()[1][nid]:
                continue
            data = tree.isa(nid)
            rmw_slots = _rmw_operand_slots(data)
            rmw_regions = _leaf_operand_regions(tree, nid, tensor, rmw_only=True)
            rmw_invariant = bool(rmw_regions) and not any(
                loop.loop_var in expr_variables(lo) for region in rmw_regions for lo, _w in region.ranges
            )
            has_invariant_rmw |= rmw_invariant
            has_enclosed_init |= (
                rmw_invariant
                and isinstance(data.kwargs.get("accumulate"), tuple)
                and all(
                    _access_invariant_across(tree, nid, loop.loop_var, region.tensor)
                    for region in data.operand_bindings.values()
                )
            )
            if rmw_regions:
                continue
            for slot, region in data.operand_bindings.items():
                if (
                    region.tensor == tensor
                    and slot not in rmw_slots
                    and slot not in data.op_cls.INPUT_OPERANDS
                    and not any(loop.loop_var in expr_variables(lo) for lo, _w in region.ranges)
                ):
                    has_enclosed_init = True
        carried = has_invariant_rmw and not has_enclosed_init
        cache[key] = carried
    return carried


_PROMOTED_SPANS: WeakKeyDictionary[Dependency, dict[tuple[int, str], float]] = WeakKeyDictionary()
_LOOP_SCOPES: WeakKeyDictionary[Dependency, dict[int, tuple[tuple[int, str], ...]]] = WeakKeyDictionary()
_CHILD_ORDER: WeakKeyDictionary[Dependency, dict[int, tuple[tuple[int, ...], dict[int, int]]]] = WeakKeyDictionary()
_INSERTION_BOUNDS: WeakKeyDictionary[
    Dependency, dict[int, tuple[tuple[tuple[str, float, int], ...], tuple[tuple[str, float, int], ...]]]
] = WeakKeyDictionary()
