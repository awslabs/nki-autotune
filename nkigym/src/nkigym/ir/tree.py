"""Canonical schedule tree for an ``f_nkigym`` kernel, backed by ``networkx``.

The tree is stored as an ``nx.DiGraph`` where every node is a stable
integer id and the payload lives at ``graph.nodes[id]["data"]``. Three
payload dataclasses discriminate the node kind:

* :class:`RootNode` — dummy root of the forest.
* :class:`ForNode` — a trip loop (``trip = extent // tile_size``).
* :class:`ISANode` — a single NKI instruction with operands split into
  read / write / read-modify-write sets, plus a ``tensorize_sizes``
  map carrying the per-axis tile width lowered onto the ISA call's
  slice width.

:class:`KernelTree` wraps the graph with a small traversal surface
(``children``, ``parent``, ``ancestors``, ``descendants``, ``leaves``,
``preorder``) so downstream atoms don't have to touch ``networkx``
directly. :func:`build_initial_tree` walks an ``@nkigym_kernel``
callable via :func:`nkigym.ir.dimension_analysis.analyze_dimensions`,
lays alloc leaves at the forest root in declaration order, then hangs
one per-op loop nest per compute op. Each nest contributes one
:class:`ForNode` per axis with ``trip = extent // tile_size``; the
terminal :class:`ISANode` records ``tensorize_sizes[dim] = tile_size``
(``tile_size = op_cls.MAX_TILE_SIZE[abstract]`` when set, else the
full axis extent — so unbounded axes emit ``Loop trip=1`` with the
full extent carried on the ISA node).
Visualization helpers live in :mod:`nkigym.ir.tree_visualize`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from nkigym.ir.dimension_analysis import TensorDims, _AnalysisResult, _OpRecord
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole, NKIOp


@dataclass(frozen=True, kw_only=True)
class RootNode:
    """Dummy root payload."""


@dataclass(frozen=True, kw_only=True)
class ForNode:
    """Trip-loop payload.

    Attributes:
        dim: Concrete dim id (e.g. ``"d0"``).
        trip: Loop trip count (``extent // tile_size``).
        loop_type: Per-op axis classification
            (``PARALLEL`` / ``SEQUENTIAL`` / ``ACCUMULATION``).
    """

    dim: str
    trip: int
    loop_type: AxisRole


@dataclass(frozen=True, kw_only=True)
class ISANode:
    """NKI-instruction payload.

    Attributes:
        op_cls: The :class:`NKIOp` subclass (e.g. ``NKILoad``).
        reads: Tensor names from slots in ``op_cls.INPUT_OPERANDS``.
        writes: Tensor names from slots that are neither input nor RMW.
        rmw: Tensor names from slots in ``op_cls.RMW_OPERANDS``.
        tensorize_sizes: Per-axis tile width (``abstract_axis → tile_size``)
            lowered onto the ISA call's slice width. Keys mirror the
            ``axis_map`` keys; for :class:`NKIAlloc` leaves they record
            the per-axis tile sized to ``NKIAlloc.MAX_TILE_SIZE`` (or
            full extent when unbounded).
        axis_map: ``abstract_axis → concrete_dim`` (e.g. ``{"K": "d0"}``).
            Render consults this alongside ``op_cls.OPERAND_AXES`` to
            resolve each slot's axis labels to concrete dim ids. For
            :class:`NKIAlloc` leaves it zips ``OPERAND_AXES["dst"]`` to
            the tensor's ``dim_ids``.
        kwargs: Non-operand call kwargs captured from the tracer
            (e.g. ``{"value": 0.0}`` for ``NKIMemset``,
            ``{"op": "rsqrt", "scale": 1.0}`` for ``NKIActivation``).
            Empty for :class:`NKIAlloc` leaves.
        location: Memory location (``"shared_hbm"`` / ``"sbuf"`` / ``"psum"``)
            for :class:`NKIAlloc` leaves; ``None`` for compute ops.
        dtype: Declared dtype (``"float32"`` / ``"float16"`` / ``"bfloat16"``)
            for :class:`NKIAlloc` leaves; ``None`` for compute ops.
    """

    op_cls: type[NKIOp]
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    rmw: tuple[str, ...] = ()
    tensorize_sizes: dict[str, int] = field(default_factory=dict)
    axis_map: dict[str, str] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    location: str | None = None
    dtype: str | None = None


NodeData = RootNode | ForNode | ISANode


class KernelTree:
    """Schedule tree stored as an ``nx.DiGraph`` of integer node ids.

    Edges point parent → child. Child order is the networkx
    successor order (insertion order on ``DiGraph``), which matches
    source order because children are added sequentially.

    Attributes:
        graph: The underlying ``nx.DiGraph``. Node payloads live at
            ``graph.nodes[nid]["data"]``.
        root: Node id of the forest root (a :class:`RootNode`).
    """

    def __init__(self) -> None:
        """Initialise an empty tree."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self._next_id: int = 0
        self.root: int = self.add_node(RootNode())

    def add_node(self, data: NodeData, parent: int | None = None) -> int:
        """Add a node with ``data`` as payload; return the new node id."""
        nid = self._next_id
        self._next_id += 1
        self.graph.add_node(nid, data=data)
        if parent is not None:
            self.graph.add_edge(parent, nid)
        return nid

    @property
    def num_nodes(self) -> int:
        """Total node count in the underlying graph (includes the root)."""
        return self.graph.number_of_nodes()

    def data(self, nid: int) -> NodeData:
        """Return the payload attached to node ``nid``."""
        return self.graph.nodes[nid]["data"]

    def children(self, nid: int) -> list[int]:
        """Return the ordered list of direct children of ``nid``."""
        return list(self.graph.successors(nid))

    def parent(self, nid: int) -> int | None:
        """Return the parent of ``nid`` (``None`` for the root)."""
        preds = list(self.graph.predecessors(nid))
        if not preds:
            return None
        if len(preds) > 1:
            raise ValueError(f"Node {nid} has multiple parents: {preds}")
        return preds[0]

    def ancestors(self, nid: int) -> list[int]:
        """Return ancestors of ``nid``, root-first."""
        chain: list[int] = []
        cur = self.parent(nid)
        while cur is not None:
            chain.append(cur)
            cur = self.parent(cur)
        chain.reverse()
        return chain

    def descendants(self, nid: int) -> set[int]:
        """Return the set of all transitive descendants of ``nid``."""
        return set(nx.descendants(self.graph, nid))

    def preorder(self, nid: int | None = None) -> Iterator[int]:
        """Yield node ids in pre-order DFS from ``nid`` (default: root)."""
        start = self.root if nid is None else nid
        yield from nx.dfs_preorder_nodes(self.graph, source=start)

    def leaves(self, nid: int | None = None) -> Iterator[int]:
        """Yield leaves (out-degree 0) reachable from ``nid``."""
        for m in self.preorder(nid):
            if self.graph.out_degree(m) == 0:
                yield m


def build_initial_tree(analysis: _AnalysisResult) -> KernelTree:
    """Build the canonical schedule tree from an :class:`_AnalysisResult`.

    Iterates ``analysis.ops`` uniformly — each :class:`_OpRecord`
    (including :class:`NKIAlloc` records prepended by the tracer)
    becomes one call to :func:`_attach_op_subtree`. Alloc leaves sit as
    direct children of the root (no surrounding loops); compute ops get
    a per-axis loop nest outermost-to-innermost in axis_map order.
    """
    tree = KernelTree()
    for op in analysis.ops:
        _attach_op_subtree(tree, op, analysis.dim_sizes, analysis.tensors)
    return tree


def _attach_op_subtree(
    tree: KernelTree, op: _OpRecord, dim_sizes: dict[str, int], tensors: dict[str, TensorDims]
) -> None:
    """Attach one op subtree under ``tree.root``.

    Builds the per-axis loop chain (skipped for :class:`NKIAlloc`) and
    the terminal :class:`ISANode`. Operand slots are split into reads /
    writes / rmw via ``op_cls.INPUT_OPERANDS`` and ``op_cls.RMW_OPERANDS``.
    Per-axis tile widths come from ``op_cls.MAX_TILE_SIZE`` (full extent
    when ``None``). For :class:`NKIAlloc` leaves the declared
    ``location`` and ``dtype`` are copied from the allocated
    :class:`TensorDims` onto the leaf so renderers don't have to
    round-trip through ``ir.tensors``.
    """
    reads: list[str] = []
    writes: list[str] = []
    rmw: list[str] = []
    for slot, tensor_name in op.operand_names.items():
        if slot in op.op_cls.INPUT_OPERANDS:
            reads.append(tensor_name)
        elif slot in op.op_cls.RMW_OPERANDS:
            rmw.append(tensor_name)
        else:
            writes.append(tensor_name)
    emit_loops = op.op_cls is not NKIAlloc
    parent = tree.root
    tensorize_sizes: dict[str, int] = {}
    for abstract, concrete in op.axis_map.items():
        extent = dim_sizes[concrete]
        max_tile = op.op_cls.MAX_TILE_SIZE.get(abstract)
        tile = extent if max_tile is None else max_tile
        tensorize_sizes[abstract] = tile
        if emit_loops:
            role = op.op_cls.AXIS_ROLES.get(abstract, AxisRole.PARALLEL)
            parent = tree.add_node(ForNode(dim=concrete, trip=extent // tile, loop_type=role), parent=parent)
    is_alloc = op.op_cls is NKIAlloc
    location = tensors[writes[0]].location if is_alloc else None
    dtype = tensors[writes[0]].dtype if is_alloc else None
    tree.add_node(
        ISANode(
            op_cls=op.op_cls,
            reads=tuple(reads),
            writes=tuple(writes),
            rmw=tuple(rmw),
            location=location,
            dtype=dtype,
            tensorize_sizes=tensorize_sizes,
            axis_map=dict(op.axis_map),
            kwargs=dict(op.kwargs),
        ),
        parent=parent,
    )


__all__ = ["ForNode", "ISANode", "KernelTree", "NodeData", "RootNode", "build_initial_tree"]
