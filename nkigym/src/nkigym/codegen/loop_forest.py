"""``LoopForest`` IR — tree-level analysis surface for structural kernel rewrites.

Each `OpGraph` op lowers to one `LoopForest` tree. A tree's interior
`LoopNode`s describe the loop structure emitted at render time; its
leaves are `BodyLeaf` markers that name which op (and which phase of
that op, for multi-phase ops like matmul) runs at that tree position.
Transforms such as :class:`FuseOuterLoop` operate on the forest as data
rather than on source text.
"""

from dataclasses import dataclass, field

from nkigym.codegen.graph import OpGraph, ParsedOp
from nkigym.ops.base import AxisRole


@dataclass
class BodyLeaf:
    """Marks where an op (or an op phase) runs inside a loop nest.

    Attributes:
        op_idx: Index into ``OpGraph.ops`` of the op this leaf represents.
        phase: Phase name for multi-phase ops. Single-phase ops use the
            default ``"main"``. Matmul phases: ``"psum_init"``,
            ``"compute"``, ``"drain"``. ActivationReduce phases:
            ``"reducer_init"``, ``"reduce_step"``, ``"post_op"``.
    """

    op_idx: int
    phase: str = "main"


@dataclass
class LoopNode:
    """A single loop at one tree depth.

    Attributes:
        dim_id: Concrete ``OpGraph.dims`` key this loop iterates.
        trip_count: Iteration count (``num_tiles(d)`` for a "block" tier,
            ``1`` for a "tile" tier in the 2N-per-dim canonical form;
            any divisor of ``num_tiles(d)`` under structural transforms).
        role: ``AxisRole`` for this op's use of ``dim_id``. After
            fusion the merged ``LoopNode``'s role is ``PARALLEL`` by
            construction (fusion requires both sides PARALLEL).
        children: Nested ``LoopNode``s and/or terminal ``BodyLeaf``s, in
            emission order.
    """

    dim_id: str
    trip_count: int
    role: AxisRole
    children: "list[LoopNode | BodyLeaf]" = field(default_factory=list)


LoopForest = list[LoopNode | BodyLeaf]
"""A list of root-level entries, one per op, in program order."""


def check_invariant(
    forest: LoopForest,
    num_tiles: dict[str, int],
    op_touched: dict[int, tuple[str, ...]],
    phase_touched: dict[tuple[int, str], tuple[str, ...]] | None = None,
) -> None:
    """Validate the per-dim product invariant on every ``BodyLeaf``.

    Walks the forest from each root; at every ``BodyLeaf`` verifies that
    for each ``dim_id d`` in the leaf's op's ``touched_dims``, the
    product of ancestor ``LoopNode.trip_count`` where ``dim_id == d``
    equals ``num_tiles[d]``.

    Args:
        forest: The forest to validate.
        num_tiles: Maps dim_id to expected total tile count.
        op_touched: Maps op_idx to its full ``touched_dims``.
        phase_touched: Optional override — maps ``(op_idx, phase)`` to the
            subset of dims that phase actually touches. For multi-phase
            ops where different phases access different dims (e.g. matmul's
            psum_init/drain don't touch K), this refines the check.

    Raises:
        ValueError: A body leaf violates the invariant; the message
            names the offending dim and op index.
    """
    if phase_touched is None:
        phase_touched = {}
    for entry in forest:
        _check_node(entry, ancestor_trips={}, num_tiles=num_tiles, op_touched=op_touched, phase_touched=phase_touched)


def _check_node(
    node: LoopNode | BodyLeaf,
    ancestor_trips: dict[str, list[int]],
    num_tiles: dict[str, int],
    op_touched: dict[int, tuple[str, ...]],
    phase_touched: dict[tuple[int, str], tuple[str, ...]],
) -> None:
    """Recursive helper for :func:`check_invariant`."""
    if isinstance(node, BodyLeaf):
        """Phase-specific dims override the full op touched_dims if present."""
        key = (node.op_idx, node.phase)
        touched = phase_touched.get(key, op_touched.get(node.op_idx, ()))
        for d in touched:
            trips = ancestor_trips.get(d, [])
            if not trips:
                raise ValueError(
                    f"BodyLeaf(op_idx={node.op_idx}, phase={node.phase!r}) references dim "
                    f"{d!r} but no ancestor LoopNode iterates it"
                )
            product = 1
            for t in trips:
                product *= t
            expected = num_tiles[d]
            if product != expected:
                raise ValueError(
                    f"BodyLeaf(op_idx={node.op_idx}, phase={node.phase!r}) dim {d!r}: "
                    f"ancestor trip product {product} != num_tiles {expected}"
                )
        return
    ancestor_trips.setdefault(node.dim_id, []).append(node.trip_count)
    for child in node.children:
        _check_node(child, ancestor_trips, num_tiles, op_touched, phase_touched)
    ancestor_trips[node.dim_id].pop()


def build_canonical_forest(op_graph: OpGraph) -> LoopForest:
    """Produce the canonical forest — one tree per op, in program order.

    Each op's tree is a 2N-deep chain over its ``touched_dims``: for dim
    ``d_k`` we emit a ``LoopNode(d_k, num_tiles(d_k))`` followed by a
    nested ``LoopNode(d_k, 1)`` ("block" tier then "tile" tier). At the
    deepest point, multi-phase ops place phase leaves per op-class
    rules; single-phase ops place one ``BodyLeaf(op_idx, "main")``.
    """
    return [_build_tree(op, op_graph) for op in op_graph.ops]


def compute_phase_touched(op_graph: OpGraph) -> dict[tuple[int, str], tuple[str, ...]]:
    """Build the phase-specific touched-dims map for multi-phase ops.

    Returns a mapping from ``(op_idx, phase)`` to the dims that phase
    actually touches. Single-phase ops (phase="main") are omitted — the
    invariant checker falls back to ``op_touched`` for those.
    """
    result: dict[tuple[int, str], tuple[str, ...]] = {}
    for op in op_graph.ops:
        phase_fn = _PHASE_DIMS.get(op.op_cls.__name__)
        if phase_fn is not None:
            phase_map = phase_fn(op)
            for phase, dims in phase_map.items():
                result[(op.idx, phase)] = dims
    return result


def _build_tree(op: ParsedOp, op_graph: OpGraph) -> LoopNode:
    """Build the 2N-per-dim chain for ``op`` with phase leaves at the tip."""
    deepest_children = _build_leaves(op, op_graph)
    wrap_dims = _dims_to_wrap(op)
    return _wrap_dims(wrap_dims, op, op_graph, deepest_children)


def _dims_to_wrap(op: ParsedOp) -> tuple[str, ...]:
    """Return the dims the outer wrapper should build around the leaves.

    Multi-phase builders may handle some interior dims themselves (e.g.
    matmul builds K internally). For those builders, the dims they
    consume are dropped from the outer wrap.
    """
    interior_fn = _BUILDER_INTERIOR_DIMS.get(op.op_cls.__name__)
    skip = interior_fn(op) if interior_fn is not None else set()
    return tuple(d for d in op.touched_dims if d not in skip)


def _wrap_dims(
    dims: tuple[str, ...], op: ParsedOp, op_graph: OpGraph, inner_children: list[LoopNode | BodyLeaf]
) -> LoopNode:
    """Wrap ``inner_children`` in a 2N-per-dim chain over ``dims``."""
    if not dims:
        raise ValueError(f"Op {op.idx}: cannot build tree — no touched_dims to wrap")
    node_children: list[LoopNode | BodyLeaf] = inner_children
    for d in reversed(dims):
        role = op.dim_role[d]
        num_t = op_graph.dims[d].num_tiles
        tile_node = LoopNode(dim_id=d, trip_count=1, role=role, children=node_children)
        block_node = LoopNode(dim_id=d, trip_count=num_t, role=role, children=[tile_node])
        node_children = [block_node]
    head = node_children[0]
    assert isinstance(head, LoopNode)
    return head


def _build_leaves(op: ParsedOp, op_graph: OpGraph) -> list[LoopNode | BodyLeaf]:
    """Return the deepest-point children list for ``op``'s tree.

    Dispatch on op-class name — single-phase ops return a single
    ``[BodyLeaf(op.idx, "main")]``; multi-phase ops (matmul,
    activation_reduce) receive custom builders added in later tasks
    (registered via :data:`_LEAF_BUILDERS`).
    """
    builder = _LEAF_BUILDERS.get(op.op_cls.__name__, _build_leaves_default)
    return builder(op, op_graph)


def _build_leaves_default(op: ParsedOp, op_graph: OpGraph) -> list[LoopNode | BodyLeaf]:
    """Single-phase default: one ``BodyLeaf(op_idx, 'main')``."""
    _ = op_graph
    return [BodyLeaf(op_idx=op.idx, phase="main")]


def _build_leaves_matmul(op: ParsedOp, op_graph: OpGraph) -> list[LoopNode | BodyLeaf]:
    """Matmul: ``[psum_init leaf, <K chain ending in compute leaf>, drain leaf]``.

    The outer M and N dims are consumed by ``_wrap_dims``. The K dim is
    handled here so the body placement mirrors the physical kernel:
    PSUM init lives outside K, ``nc_matmul`` fires inside K, drain
    runs after K closes.
    """
    k_dim = op.axis_map["K"]
    k_role = op.dim_role[k_dim]
    num_k = op_graph.dims[k_dim].num_tiles
    compute_leaf = BodyLeaf(op_idx=op.idx, phase="compute")
    k_tile = LoopNode(dim_id=k_dim, trip_count=1, role=k_role, children=[compute_leaf])
    k_block = LoopNode(dim_id=k_dim, trip_count=num_k, role=k_role, children=[k_tile])
    return [BodyLeaf(op_idx=op.idx, phase="psum_init"), k_block, BodyLeaf(op_idx=op.idx, phase="drain")]


_LEAF_BUILDERS: dict = {}
"""Populated by multi-phase builders in Tasks B3 / B4 (matmul, activation_reduce)."""

_BUILDER_INTERIOR_DIMS: dict = {}
"""Maps op-class name to a callable `(op) -> set[dim_id]` of dims that the
custom leaf builder handles internally (to be skipped by the outer wrap).
Populated in Tasks B3 / B4."""


def _phase_dims_matmul(op: ParsedOp) -> dict[str, tuple[str, ...]]:
    """Return the dims each matmul phase touches.

    psum_init and drain run outside K; compute runs inside K.
    """
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    k_dim = op.axis_map["K"]
    return {"psum_init": (m_dim, n_dim), "compute": (m_dim, n_dim, k_dim), "drain": (m_dim, n_dim)}


def _build_leaves_activation_reduce(op: ParsedOp, op_graph: OpGraph) -> list[LoopNode | BodyLeaf]:
    """ActivationReduce: ``[reducer_init leaf, <F chain ending in reduce_step leaf>, post_op leaf?]``.

    The outer P dim is consumed by ``_wrap_dims``. The F dim is handled
    here: an ``F-block / F-tile / BodyLeaf(reduce_step)`` chain sits
    between ``reducer_init`` and the optional ``post_op`` leaf. The
    ``post_op`` leaf is included only when ``op.op_kwargs["post_op"]``
    is not ``None``.
    """
    f_dim = op.axis_map["F"]
    f_role = op.dim_role[f_dim]
    num_f = op_graph.dims[f_dim].num_tiles
    reduce_leaf = BodyLeaf(op_idx=op.idx, phase="reduce_step")
    f_tile = LoopNode(dim_id=f_dim, trip_count=1, role=f_role, children=[reduce_leaf])
    f_block = LoopNode(dim_id=f_dim, trip_count=num_f, role=f_role, children=[f_tile])
    leaves: list[LoopNode | BodyLeaf] = [BodyLeaf(op_idx=op.idx, phase="reducer_init"), f_block]
    if op.op_kwargs.get("post_op") is not None:
        leaves.append(BodyLeaf(op_idx=op.idx, phase="post_op"))
    return leaves


def _phase_dims_activation_reduce(op: ParsedOp) -> dict[str, tuple[str, ...]]:
    """Return the dims each activation_reduce phase touches.

    reducer_init and post_op run outside F; reduce_step runs inside F.
    """
    p_dim = op.axis_map["P"]
    f_dim = op.axis_map["F"]
    return {"reducer_init": (p_dim,), "reduce_step": (p_dim, f_dim), "post_op": (p_dim,)}


_LEAF_BUILDERS["NKIMatmul"] = _build_leaves_matmul
_BUILDER_INTERIOR_DIMS["NKIMatmul"] = lambda op: {op.axis_map["K"]}
_LEAF_BUILDERS["NKIActivationReduce"] = _build_leaves_activation_reduce
_BUILDER_INTERIOR_DIMS["NKIActivationReduce"] = lambda op: {op.axis_map["F"]}
_PHASE_DIMS: dict = {}
"""Maps op-class name to a callable `(op) -> dict[phase, tuple[dim_id, ...]]`
that returns the dims each phase of that op touches. Used by
:func:`check_invariant` to validate multi-phase ops correctly."""
_PHASE_DIMS["NKIMatmul"] = _phase_dims_matmul
_PHASE_DIMS["NKIActivationReduce"] = _phase_dims_activation_reduce
