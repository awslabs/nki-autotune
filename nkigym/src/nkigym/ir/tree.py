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
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import networkx as nx

from nkigym.ir.arith.expr import Expr
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


@dataclass(frozen=True, kw_only=True)
class AccessPattern:
    """Flattened multidimensional view used by one ISA operand.

    ``pattern`` stores ``(stride, extent)`` pairs in view-axis order and
    ``offset`` stores the flattened base element. The corresponding
    :class:`BufferRegion` remains the logical footprint used by dependency
    analysis.
    """

    pattern: tuple[tuple[Expr, Expr], ...]
    offset: Expr


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
    access_patterns: dict[str, AccessPattern] = field(default_factory=dict)


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
        storage_dtype: Optional physical allocation dtype. ``None`` uses
            ``dtype``.
        versions: pipeline buffer-version count (default 1).
        list_len: list-of-tiles count (default 1).
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    location: str
    storage_dtype: str | None = None
    versions: int = 1
    """Pipeline buffer-version count. 1 = single instance (renders
    byte-identically to today). >1 multiplies the tile (middle) dim of
    each list allocation so the renderer's ``loop_var % versions`` rotation
    addresses distinct slots. Set by SoftwarePipeline (use_stage − def_stage
    + 1); left 1 everywhere else."""
    list_len: int = 1
    """List-of-tiles count. 1 = a single packed ``nl.ndarray`` (renders
    byte-identically to today). >1 splits the buffer into a Python LIST of
    ``list_len`` separate ndarrays, each :meth:`per_tile_physical_shape`, indexed
    by a leading list subscript at the call site. Each list entry contains its
    logical tiles for every :attr:`versions` slot. Set by the BufferLayout transform;
    left 1 everywhere else."""

    def _on_chip_shape(self) -> tuple[int, int]:
        """Return the logical ``(leading, free)`` shape for an on-chip buffer."""
        if self.location == "shared_hbm":
            raise AssertionError(f"{self.name}: shared_hbm has no on-chip tile shape")
        if len(self.shape) == 1:
            leading, free = self.shape[0], 1
        elif len(self.shape) == 2:
            leading, free = self.shape
        else:
            raise AssertionError(f"{self.name}: SBUF/PSUM buffer expects a 1D or 2D logical shape; got {self.shape}")
        partition = min(leading, PARTITION_DIM)
        if leading % partition != 0:
            raise AssertionError(f"{self.name}: leading extent {leading} cannot use partition extent {partition}")
        return leading, free

    def partition_extent(self) -> int:
        """Return the physical partition width of one on-chip tile."""
        leading, _free = self._on_chip_shape()
        return min(leading, PARTITION_DIM)

    def logical_tile_count(self) -> int:
        """Return the number of logical partition tiles before versioning."""
        leading, _free = self._on_chip_shape()
        return leading // self.partition_extent()

    def tiles_per_list(self) -> int:
        """Return logical partition tiles stored in each list entry."""
        logical_tiles = self.logical_tile_count()
        if self.list_len < 1 or logical_tiles % self.list_len != 0:
            raise AssertionError(
                f"{self.name}: list_len {self.list_len} must divide logical tile count {logical_tiles}"
            )
        return logical_tiles // self.list_len

    def physical_shape(self) -> tuple[int, ...]:
        """Return the shape ``nl.ndarray`` actually allocates for this buffer.

        ``shared_hbm`` buffers keep their logical shape. ``sbuf`` and
        ``psum`` buffers expand to the 3D NeuronCore layout
        ``(128, num_p_tiles, F_contig)``. A logical vector ``(P,)`` uses
        ``F_contig=1``. The partition axis is fixed at 128 and the leading
        logical extent folds into the tile count. This is the single source
        of truth shared by the renderer
        (:func:`nkigym.codegen.body._emit_alloc`) and buffer transforms.
        """
        if self.location == "shared_hbm":
            return self.shape
        _leading, free = self._on_chip_shape()
        return (self.partition_extent(), self.logical_tile_count() * self.versions, free)

    def per_tile_physical_shape(self) -> tuple[int, ...]:
        """Return the ndarray shape of one entry in this buffer's allocation list.

        The list-of-tiles form (:attr:`list_len` > 1) allocates ``list_len``
        separate ndarrays, each this shape. The middle dimension contains this
        list entry's logical tiles for every pipeline version:
        ``tiles_per_list * versions``. Identity when ``list_len == 1``.
        """
        if self.list_len == 1:
            return self.physical_shape()
        if self.location == "shared_hbm":
            raise AssertionError(f"{self.name}: shared_hbm has no tile axis to split (list_len must be 1)")
        partition, _total_tiles, free = self.physical_shape()
        return (partition, self.tiles_per_list() * self.versions, free)

    def physical_dtype(self) -> str:
        """Return the dtype ``nl.ndarray`` actually allocates for this buffer.

        Most buffers use their logical :attr:`dtype`. Producers whose hardware
        destination differs set :attr:`storage_dtype`; notably matmul uses an
        fp32 PSUM accumulator while ``nc_transpose`` preserves its input dtype.
        """
        return self.storage_dtype if self.storage_dtype is not None else self.dtype


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


NodeData = BlockNode | ForNode | ISANode
_NodeT = TypeVar("_NodeT", BlockNode, ForNode, ISANode)


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

    def restore_next_id(self, next_id: int) -> None:
        """Restore the allocator after removing every node added since ``next_id``."""
        remaining = [nid for nid in self.graph.nodes if nid >= next_id]
        if remaining:
            raise ValueError(f"cannot restore next node id to {next_id}; live nodes remain: {remaining}")
        self._next_id = next_id

    @property
    def num_nodes(self) -> int:
        """Total node count in the underlying graph (includes the root)."""
        return self.graph.number_of_nodes()

    @property
    def next_node_id(self) -> int:
        """Return the node id that the next :meth:`add_node` call will allocate."""
        return self._next_id

    def data(self, nid: int) -> NodeData:
        """Return the payload attached to node ``nid``."""
        return self.graph.nodes[nid]["data"]

    def _expect_data(self, nid: int, expected: type[_NodeT]) -> _NodeT:
        """Return node data after validating its concrete payload type."""
        data = self.data(nid)
        if not isinstance(data, expected):
            raise TypeError(f"node {nid} is {type(data).__name__}, expected {expected.__name__}")
        return data

    def block(self, nid: int) -> BlockNode:
        """Return node ``nid`` as a :class:`BlockNode`, or raise :class:`TypeError`."""
        return self._expect_data(nid, BlockNode)

    def loop(self, nid: int) -> ForNode:
        """Return node ``nid`` as a :class:`ForNode`, or raise :class:`TypeError`."""
        return self._expect_data(nid, ForNode)

    def isa(self, nid: int) -> ISANode:
        """Return node ``nid`` as an :class:`ISANode`, or raise :class:`TypeError`."""
        return self._expect_data(nid, ISANode)

    def children(self, nid: int) -> list[int]:
        """Return the ordered list of direct children of ``nid``."""
        return list(self.graph.successors(nid))

    def parent(self, nid: int) -> int | None:
        """Return the parent of ``nid`` (``None`` for the root)."""
        predecessors = iter(self.graph.predecessors(nid))
        parent = next(predecessors, None)
        if parent is None:
            return None
        extra_parent = next(predecessors, None)
        if extra_parent is not None:
            raise ValueError(f"Node {nid} has multiple parents: {[parent, extra_parent, *predecessors]}")
        return parent

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
        descendants = {nid}
        pending = list(self.graph.successors(nid))
        while pending:
            descendant = pending.pop()
            if descendant not in descendants:
                descendants.add(descendant)
                pending.extend(self.graph.successors(descendant))
        descendants.remove(nid)
        return descendants

    def preorder(self, nid: int | None = None) -> Iterator[int]:
        """Yield node ids in pre-order DFS from ``nid`` (default: root)."""
        start = self.root if nid is None else nid
        pending = [start]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            yield current
            pending.extend(reversed(tuple(self.graph.successors(current))))

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
