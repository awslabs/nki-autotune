"""Canonical schedule tree for an ``f_nkigym`` kernel, backed by ``networkx``.

The tree is stored as an ``nx.DiGraph`` where every node is a stable
integer id and the payload lives at ``graph.nodes[id]["data"]``. Payload
dataclasses discriminate the node kind:

* :class:`BlockNode` — TVM-style schedulable unit owning iter_vars,
  declared reads / writes, and ``alloc_buffers``.
* :class:`ForNode` — a loop binding to (part of) a block iter_var.
* :class:`ISANode` — a single NKI instruction.

:class:`IterVar`, :class:`BufferRegion`, and :class:`Buffer` are
sub-payloads carried on :class:`BlockNode` and :class:`ISANode`.

:class:`KernelTree` wraps the graph with a small traversal surface
(``children``, ``parent``, ``ancestors``, ``descendants``, ``leaves``,
``preorder``, ``blocks``) so downstream atoms don't have to touch
``networkx`` directly. :func:`build_initial_tree` walks an
``@nkigym_kernel`` callable via :func:`nkigym.ir.dimension_analysis.analyze_dimensions`.
Visualization helpers live in :mod:`nkigym.ir.tree_visualize`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from nkigym.ir.arith.expr import Expr, format_expr
from nkigym.ir.dimension_analysis import _AnalysisResult
from nkigym.ops.base import AxisRole, NKIOp

PARTITION_DIM = 128
"""NeuronCore SBUF/PSUM partition-axis size. The single source of truth
for the 128-partition layout, shared by the canonical region builder,
the codegen renderer, the interval/overlap analysis, and
:meth:`Buffer.physical_shape`. This is the hardware partition dimension,
distinct from any per-op ``MIN_TILE_SIZE``/``MAX_TILE_SIZE`` cap."""


@dataclass(frozen=True, kw_only=True)
class ForNode:
    """Loop binding to one (or part of one) :class:`BlockNode` iter_var.

    Multiple same-axis ``ForNode``s above one block — the result of
    :class:`Split` — bind the iter_var via the affine combination
    encoded in the enclosing block's ``iter_values``.

    Attributes:
        loop_var: symbolic name (e.g. ``"i_M_outer"``).
        extent: loop trip count.
    """

    loop_var: str
    extent: int

    def label(self) -> str:
        """Return ``Loop <loop_var> extent=<extent>``."""
        return f"Loop {self.loop_var} extent={self.extent}"


@dataclass(frozen=True, kw_only=True)
class ISANode:
    """Single ISA call.

    Attributes:
        op_cls: :class:`NKIOp` subclass.
        operand_bindings: per-slot :class:`BufferRegion` in the
            enclosing :class:`BlockNode`'s iter_var space.
        kwargs: non-operand call kwargs (e.g. ``{"value": 0.0}`` for
            :class:`NKIMemset`).
    """

    op_cls: type[NKIOp]
    operand_bindings: dict[str, BufferRegion] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        """Return the op name plus per-slot region labels and kwargs, newline-separated."""
        lines: list[str] = [f"nisa.{self.op_cls.NAME}"]
        if self.operand_bindings:
            bindings = ", ".join(f"{slot}={region.label()}" for slot, region in self.operand_bindings.items())
            lines.append(f"bindings=({bindings})")
        if self.kwargs:
            lines.append(f"kwargs={self.kwargs}")
        return "\n".join(lines)


@dataclass(frozen=True, kw_only=True)
class IterVar:
    """Per-block iteration variable.

    Attributes:
        axis: abstract axis name (``"M"``, ``"K"``, ``"P"``, ...).
        dom: half-open extent ``(lo, hi)``.
        role: ``PARALLEL`` (TVM ``kDataPar``) / ``ACCUMULATION``
            (``kCommReduce``) / ``SEQUENTIAL`` (``kOrdered``).
    """

    axis: str
    dom: tuple[int, int]
    role: AxisRole

    def label(self) -> str:
        """Return ``axis(ROL lo..hi)`` with the role abbreviated to 3 letters."""
        return f"{self.axis}({self.role.name[:3]} {self.dom[0]}..{self.dom[1]})"


@dataclass(frozen=True, kw_only=True)
class Buffer:
    """Buffer declaration on an enclosing :class:`BlockNode`.

    Replaces the standalone :class:`NKIAlloc` ISA leaf. The lifetime is
    bounded by the declaring block.

    Attributes:
        name: tensor name.
        shape: per-axis extent.
        dtype: ``"float32"`` / ``"float16"`` / ``"bfloat16"``.
        location: ``"shared_hbm"`` / ``"sbuf"`` / ``"psum"``.
        versions: pipeline buffer-version count (default 1).
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    location: str
    versions: int = 1
    """Pipeline buffer-version count. 1 = single instance (renders
    byte-identically to today). >1 multiplies the tile (middle) dim of
    physical_shape so the renderer's ``loop_var % versions`` rotation
    addresses distinct slots. Set by SoftwarePipeline (use_stage − def_stage
    + 1); left 1 everywhere else."""

    def physical_shape(self) -> tuple[int, ...]:
        """Return the shape ``nl.ndarray`` actually allocates for this buffer.

        ``shared_hbm`` buffers keep their 2D logical shape. ``sbuf`` and
        ``psum`` buffers expand to the 3D NeuronCore layout
        ``(128, num_p_tiles, F_contig)`` — the partition axis is fixed at
        128 and the leading logical extent folds into the tile count. This
        is the single source of truth shared by the renderer
        (:func:`nkigym.codegen.body._emit_alloc`) and the tree
        visualization, so the two never drift.
        """
        if self.location == "shared_hbm":
            return self.shape
        if len(self.shape) != 2:
            raise AssertionError(f"{self.name}: SBUF/PSUM buffer expects a 2D logical shape; got {self.shape}")
        leading, free = self.shape
        if leading % PARTITION_DIM != 0:
            raise AssertionError(f"{self.name}: leading extent {leading} must be a multiple of {PARTITION_DIM}")
        return (PARTITION_DIM, (leading // PARTITION_DIM) * self.versions, free)

    def physical_dtype(self) -> str:
        """Return the dtype ``nl.ndarray`` actually allocates for this buffer.

        ``psum`` buffers are always allocated ``float32`` — the matmul HW
        accumulates at fp32 regardless of the logical dtype the value
        carries. Every other location uses the logical :attr:`dtype`. This
        is the physical override paired with :meth:`physical_shape`.
        """
        if self.location == "psum":
            return "float32"
        return self.dtype

    def label(self) -> str:
        """Return ``name (physical_shape) dtype@location`` on one line.

        Shows the physical allocation shape (3D for sbuf/psum, 2D for
        shared_hbm) so the visualization matches the rendered kernel.
        """
        shape_str = ", ".join(str(extent) for extent in self.physical_shape())
        return f"{self.name} ({shape_str}) {self.dtype}@{self.location}"


@dataclass(frozen=True, kw_only=True)
class BufferRegion:
    """Affine half-open region of a buffer, expressed in iter_var ``Var``s.

    Attributes:
        tensor: tensor name (key into the kernel's buffers).
        ranges: one ``(lo, hi)`` pair per axis, in iter_var-Var space.
            For a single-element access, ``hi`` is ``lo + 1``; for a
            tile, ``hi`` is ``lo + tile_size``.
    """

    tensor: str
    ranges: tuple[tuple[Expr, Expr], ...]

    def label(self) -> str:
        """Return ``tensor[lo : +width, ...]`` from the stored ``(lo, width)`` ranges."""
        axes = ", ".join(f"{format_expr(lo)} : +{format_expr(width)}" for lo, width in self.ranges)
        return f"{self.tensor}[{axes}]"


def _label_lines(items: tuple[BufferRegion | Buffer, ...], indent: int) -> str:
    """Join each item's ``label()`` onto its own line; continuation lines indented.

    Returns ``∅`` when ``items`` is empty so empty fields stay visible.
    """
    pad = "\n" + " " * indent
    result = pad.join(item.label() for item in items) if items else "∅"
    return result


@dataclass(frozen=True, kw_only=True)
class BlockNode:
    """TVM-style block — schedulable unit aligned with ``tir.SBlockNode``.

    Attributes:
        iter_vars: per-axis iter_vars owned by this block.
        iter_values: one Expr per iter_var (in iter_vars order) mapping
            surrounding ``ForNode.loop_var`` symbols to iter_var values.
        reads: declared read regions in iter_var space.
        writes: declared write regions in iter_var space.
        alloc_buffers: buffers whose lifetime is bounded by this block.
        annotations: free-form per-block metadata.
        axis_map: abstract op-axis → concrete dim bijection (see field doc).
    """

    iter_vars: tuple[IterVar, ...]
    iter_values: tuple[Expr, ...]
    reads: tuple[BufferRegion, ...]
    writes: tuple[BufferRegion, ...]
    alloc_buffers: tuple[Buffer, ...] = ()
    annotations: dict[str, Any] = field(default_factory=dict)
    axis_map: dict[str, str] = field(default_factory=dict)
    """Abstract op-axis (``P``/``F``/``K``/``M``/``N``) → concrete dim
    (``d0``/``d1``...). The per-block bijection between an op's
    ``OPERAND_AXES`` names and the block's concrete iter_var axes. Set once
    at canonical build (from the op record) and carried unchanged through
    every transform (no transform renames a concrete dim). Lets the
    tensorize-Split path translate a concrete ``target_axis`` to the
    abstract name ``OPERAND_AXES`` is keyed by. Empty for the synthetic
    root block and hand-built blocks with no operand axes."""

    def label(self) -> str:
        """Return a multi-line summary of all six fields; empty fields show as ∅."""
        if self.iter_vars:
            iv_line = " ".join(iv.label() for iv in self.iter_vars)
            val_line = "  ".join(f"{iv.axis}={format_expr(val)}" for iv, val in zip(self.iter_vars, self.iter_values))
        else:
            iv_line = "∅"
            val_line = "∅"
        lines = [
            "BlockNode",
            f"iter_vars:   {iv_line}",
            f"iter_values: {val_line}",
            f"reads:   {_label_lines(self.reads, 9)}",
            f"writes:  {_label_lines(self.writes, 9)}",
            f"allocs:  {_label_lines(self.alloc_buffers, 9)}",
            f"annotations: {self.annotations if self.annotations else '∅'}",
        ]
        return "\n".join(lines)


NodeData = BlockNode | ForNode | ISANode


class KernelTree:
    """Schedule tree stored as an ``nx.DiGraph`` of integer node ids.

    Edges point parent → child. Child order is the networkx
    successor order (insertion order on ``DiGraph``), which matches
    source order because children are added sequentially.

    Attributes:
        graph: The underlying ``nx.DiGraph``. Node payloads live at
            ``graph.nodes[nid]["data"]``.
        root: Node id of the root block (a :class:`BlockNode`).
    """

    def __init__(self) -> None:
        """Initialise an empty tree with root BlockNode."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self._next_id: int = 0
        self.root: int = self.add_node(BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=()))

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

    def blocks(self, nid: int | None = None) -> Iterator[int]:
        """Yield ``BlockNode``-bearing nids in pre-order DFS from ``nid``.

        Convenience for transforms that walk blocks rather than ISA leaves.
        ``nid`` defaults to the root.
        """
        for m in self.preorder(nid):
            if isinstance(self.data(m), BlockNode):
                yield m


def build_initial_tree(analysis: "_AnalysisResult") -> "KernelTree":
    """Build the canonical schedule tree from an :class:`_AnalysisResult`.

    The returned tree's root is a :class:`BlockNode` (empty iter_vars/reads/writes,
    holds kernel-lifetime buffers). Per-op leaf blocks are children of the root
    block, in source order. Allocs become ``Buffer`` entries on the smallest
    enclosing block whose subtree contains every leaf that touches the buffer
    (canonical: nearly always the root block).
    """
    from nkigym.ir.canonical_build import build_canonical_blocknode_tree

    return build_canonical_blocknode_tree(analysis)


def role_of(block: BlockNode, axis: str) -> AxisRole:
    """Return the role this block assigns to ``axis``.

    Searches ``block.iter_vars`` for the entry whose ``axis`` matches.
    Raises :class:`KeyError` if the block does not declare that axis.
    """
    for iv in block.iter_vars:
        if iv.axis == axis:
            return iv.role
    raise KeyError(f"BlockNode does not declare axis {axis!r}")


__all__ = [
    "BlockNode",
    "Buffer",
    "BufferRegion",
    "ForNode",
    "ISANode",
    "IterVar",
    "KernelTree",
    "NodeData",
    "PARTITION_DIM",
    "build_initial_tree",
    "role_of",
]
