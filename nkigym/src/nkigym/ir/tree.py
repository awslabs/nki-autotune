"""Canonical schedule tree for an ``f_nkigym`` kernel, backed by ``networkx``.

The tree is stored as an ``nx.DiGraph`` where every node is a stable
integer id and the payload lives at ``graph.nodes[id]["data"]``. Four
payload dataclasses discriminate the node kind:

* :class:`RootNode` — dummy root of the forest.
* :class:`ForNode` — an outer trip loop (``trip = extent // tile_size``).
* :class:`TensorizeLoop` — the innermost tile loop (``trip = tile_size``).
  Its extent lands on the ISA call's slice width; the renderer elides
  the ``for`` header.
* :class:`ISANode` — a single NKI instruction with operands split into
  read / write / read-modify-write sets.

:class:`KernelTree` wraps the graph with a small traversal surface
(``children``, ``parent``, ``ancestors``, ``descendants``, ``leaves``,
``preorder``) so downstream atoms don't have to touch ``networkx``
directly. :func:`build_initial_tree` walks an ``@nkigym_kernel``
callable via :func:`nkigym.ir.dimension_analysis.analyze_dimensions`,
lays alloc leaves at the forest root in declaration order, then hangs
one per-op loop nest per compute op. Each nest contributes two loops
per axis: a :class:`ForNode` with ``trip = extent // tile_size`` and
a :class:`TensorizeLoop` with ``trip = tile_size`` (``tile_size =
op_cls.MAX_TILE_SIZE[abstract]`` when set, else the full extent — so
unbounded axes emit ``Loop trip=1 → TensorizeLoop trip=extent``).
:meth:`KernelTree.dump` writes the tree as a Mermaid ``flowchart TB``
source plus a rendered PNG under a caller-supplied cache directory.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from nkigym.ir._mermaid import ClassStyle, Flowchart, render_png
from nkigym.ir.dimension_analysis import DimensionAnalysis, OpAxes
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole, NKIOp

_FLOWCHART_STYLES: list[ClassStyle] = [
    ClassStyle(name="alloc", fill="#fef", stroke="#963"),
    ClassStyle(name="loop", fill="#eef", stroke="#336"),
    ClassStyle(name="tensorize", fill="#ffe", stroke="#a60"),
    ClassStyle(name="leaf", fill="#efe", stroke="#363"),
]


@dataclass(frozen=True, kw_only=True)
class RootNode:
    """Dummy root payload."""


@dataclass(frozen=True, kw_only=True)
class ForNode:
    """Outer trip-loop payload.

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
class TensorizeLoop:
    """Innermost tile-loop payload.

    The renderer elides the ``for`` header and lands ``trip`` as the
    slice width on the ISA call's buffer-access pattern. Analogous to
    TVM's ``Tensorize`` primitive (see ``docs/ir-design.md`` §7.1).

    Attributes:
        dim: Concrete dim id (e.g. ``"d0"``).
        trip: Tile size (``MAX_TILE_SIZE[abstract]`` when set, else the
            full axis extent).
        loop_type: Per-op axis classification, identical to the parent
            :class:`ForNode`'s ``loop_type``.
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
    """

    op_cls: type[NKIOp]
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    rmw: tuple[str, ...] = ()


NodeData = RootNode | ForNode | TensorizeLoop | ISANode


class KernelTree:
    """Schedule tree stored as an ``nx.DiGraph`` of integer node ids.

    Edges point parent → child. Child order is the networkx
    successor order (insertion order on ``DiGraph``), which matches
    source order because children are added sequentially.

    Attributes:
        graph: The underlying ``nx.DiGraph``. Node payloads live at
            ``graph.nodes[nid]["data"]``.
        root: Node id of the forest root (a :class:`RootNode`).
        dim_sizes: ``dim_name → extent`` (retained for display / debug).
    """

    def __init__(self, dim_sizes: dict[str, int]) -> None:
        """Initialise an empty tree carrying ``dim_sizes`` for dumping."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self._next_id: int = 0
        self.dim_sizes: dict[str, int] = dim_sizes
        self.root: int = self.add_node(RootNode())

    def add_node(self, data: NodeData, parent: int | None = None) -> int:
        """Add a node with ``data`` as payload; return the new node id."""
        nid = self._next_id
        self._next_id += 1
        self.graph.add_node(nid, data=data)
        if parent is not None:
            self.graph.add_edge(parent, nid)
        return nid

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

    def dump(self, cache_dir: str | Path) -> None:
        """Write ``tree.mmd`` and ``tree.png`` into ``cache_dir``.

        Creates ``cache_dir`` if it does not exist. The ``.mmd`` file
        holds Mermaid source; ``.png`` is rendered at ``mmdc -s 4``
        with ``--no-sandbox`` (required on gym hosts under AppArmor).
        """
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        mmd_path = cache_path / "tree.mmd"
        png_path = cache_path / "tree.png"
        mmd_path.write_text(_to_mermaid(self), encoding="utf-8")
        render_png(mmd_path, png_path)


def build_initial_tree(analysis: DimensionAnalysis) -> KernelTree:
    """Build the canonical schedule tree from a :class:`DimensionAnalysis`.

    Alloc leaves (every :class:`NKIAlloc`) sit as direct children of the
    root in declaration order. Each compute op gets its own per-axis
    loop nest; loops appear outermost-to-innermost in the op's
    ``axis_map`` iteration order (which mirrors ``OPERAND_AXES``).

    Args:
        analysis: Output of
            :func:`nkigym.ir.dimension_analysis.analyze_dimensions`.

    Returns:
        A populated :class:`KernelTree`.
    """
    tree = KernelTree(dim_sizes=analysis.dim_sizes)
    param_names = set(analysis.param_names)
    for name in analysis.tensors:
        if name in param_names:
            continue
        tree.add_node(ISANode(op_cls=NKIAlloc, writes=(name,)), parent=tree.root)
    for op in analysis.ops:
        _attach_op_subtree(tree, op)
    return tree


def _attach_op_subtree(tree: KernelTree, op: OpAxes) -> None:
    """Attach one compute-op loop nest under ``tree.root``."""
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
    parent = tree.root
    for abstract, concrete in op.axis_map.items():
        extent = tree.dim_sizes[concrete]
        max_tile = op.op_cls.MAX_TILE_SIZE.get(abstract)
        tile = extent if max_tile is None else max_tile
        role = op.op_cls.AXIS_ROLES.get(abstract, AxisRole.PARALLEL)
        parent = tree.add_node(ForNode(dim=concrete, trip=extent // tile, loop_type=role), parent=parent)
        parent = tree.add_node(TensorizeLoop(dim=concrete, trip=tile, loop_type=role), parent=parent)
    tree.add_node(ISANode(op_cls=op.op_cls, reads=tuple(reads), writes=tuple(writes), rmw=tuple(rmw)), parent=parent)


def _to_mermaid(tree: KernelTree) -> str:
    """Render ``tree`` to a Mermaid ``flowchart TB`` source string."""
    flow = Flowchart(direction="TB", styles=_FLOWCHART_STYLES)
    for nid in tree.preorder():
        node_id = f"n{nid}"
        decl, class_name = _tree_node_decl(node_id, nid, tree.data(nid))
        flow.add_node(node_id, decl, class_name)
        for child in tree.children(nid):
            flow.add_edge(node_id, f"n{child}")
    return flow.render()


def _tree_node_decl(node_id: str, nid: int, data: NodeData) -> tuple[str, str | None]:
    """Return the Mermaid declaration + class bucket for one tree node."""
    if isinstance(data, RootNode):
        return f'{node_id}(("#{nid} root"))', None
    if isinstance(data, ForNode):
        return (f'{node_id}["#{nid} Loop {data.dim} trip={data.trip}<br/>{data.loop_type.name}"]', "loop")
    if isinstance(data, TensorizeLoop):
        return (f'{node_id}["#{nid} TensorizeLoop {data.dim} trip={data.trip}<br/>{data.loop_type.name}"]', "tensorize")
    if isinstance(data, ISANode):
        return (f'{node_id}["#{nid} {_isa_label(data)}"]', "alloc" if data.op_cls is NKIAlloc else "leaf")
    raise TypeError(f"unknown node data type: {type(data).__name__}")


def _isa_label(data: ISANode) -> str:
    """Build the Mermaid node label for an :class:`ISANode` payload."""
    if data.op_cls is NKIAlloc:
        return f"alloc<br/>{data.writes[0]}"
    parts: list[str] = [data.op_cls.__name__]
    if data.reads:
        parts.append(f"reads={','.join(data.reads)}")
    if data.writes:
        parts.append(f"writes={','.join(data.writes)}")
    if data.rmw:
        parts.append(f"rmw={','.join(data.rmw)}")
    return "<br/>".join(parts)


__all__ = [
    "DimensionAnalysis",
    "ForNode",
    "ISANode",
    "KernelTree",
    "NodeData",
    "RootNode",
    "TensorizeLoop",
    "build_initial_tree",
]
