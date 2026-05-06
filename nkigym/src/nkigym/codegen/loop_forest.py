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
        reduce_op: Reducer name for ACCUMULATION loops (``"add"``,
            ``"max"``, ...). ``None`` for PARALLEL / SEQUENTIAL loops.
            Used by :class:`ReorderLoops` to detect associative-
            compatible ACC×ACC swaps.
        name: Loop variable name emitted in the rendered ``for`` header.
            Populated by :func:`build_canonical_forest` as ``f"i_{dim_id}_{k}"``
            where ``k`` counts same-dim ancestors outermost→innermost at
            build time. Preserved verbatim across :class:`FuseLoops` and
            :class:`ReorderLoops` so loop identity survives swaps. ``None``
            on raw test forests; the renderer falls back to a
            position-based name when unset.
    """

    dim_id: str
    trip_count: int
    role: AxisRole
    children: "list[LoopNode | BodyLeaf]" = field(default_factory=list)
    reduce_op: str | None = None
    name: str | None = None


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

    After construction, every ``LoopNode`` is named ``f"i_{dim_id}_{k}"``
    where ``k`` indexes same-dim ancestors root-outward at build time.
    Names are preserved verbatim across structural rewrites, so loop
    identity survives reorder and fusion.
    """
    forest = [_build_tree(op, op_graph) for op in op_graph.ops]
    for tree in forest:
        _assign_canonical_names(tree, same_dim_counts={})
    return forest


def _assign_canonical_names(node: "LoopNode | BodyLeaf", same_dim_counts: dict[str, int]) -> None:
    """Walk the tree in a root-outward DFS, naming each LoopNode.

    ``same_dim_counts[d]`` tracks how many same-dim ancestors of ``d``
    are already open on the current path; the newly visited node takes
    that as its ordinal, emits a name, then recurses with the counter
    incremented. Restoring the counter after recursion means siblings
    see the same counts the parent did (they are not each other's
    ancestors).
    """
    if isinstance(node, BodyLeaf):
        return
    k = same_dim_counts.get(node.dim_id, 0)
    node.name = f"i_{node.dim_id}_{k}"
    same_dim_counts[node.dim_id] = k + 1
    for child in node.children:
        _assign_canonical_names(child, same_dim_counts)
    same_dim_counts[node.dim_id] = k


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
    runs after K closes. The K-chain LoopNodes carry ``reduce_op="add"``
    because nc_matmul's PSUM accumulator is summation.
    """
    k_dim = op.axis_map["K"]
    k_role = op.dim_role[k_dim]
    num_k = op_graph.dims[k_dim].num_tiles
    compute_leaf = BodyLeaf(op_idx=op.idx, phase="compute")
    k_tile = LoopNode(dim_id=k_dim, trip_count=1, role=k_role, children=[compute_leaf], reduce_op="add")
    k_block = LoopNode(dim_id=k_dim, trip_count=num_k, role=k_role, children=[k_tile], reduce_op="add")
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
    """ActivationReduce Pattern 2: ``[<F-chain with reduce_step>, reduce_close]``.

    The outer P dim is consumed by ``_wrap_dims``. The F dim is handled
    here: F-block → F-tile → BodyLeaf(reduce_step) writes each tile's
    partial sum into a distinct slot of the op-local ``slot_vec``. After
    the F loop exits, ``reduce_close`` folds the slot vector via a
    single ``nisa.tensor_reduce`` into the op's ``(P, 1)`` output.

    No ``reducer_init`` phase — each ``activation_reduce`` call writes a
    distinct slot. No ``post_op`` phase — fused closures are spelled out
    as a separate ``NKIActivation`` op in the DSL.
    """
    f_dim = op.axis_map["F"]
    f_role = op.dim_role[f_dim]
    num_f = op_graph.dims[f_dim].num_tiles
    reduce_op = op.op_kwargs["reduce_op"]
    reduce_leaf = BodyLeaf(op_idx=op.idx, phase="reduce_step")
    f_tile = LoopNode(dim_id=f_dim, trip_count=1, role=f_role, children=[reduce_leaf], reduce_op=reduce_op)
    f_block = LoopNode(dim_id=f_dim, trip_count=num_f, role=f_role, children=[f_tile], reduce_op=reduce_op)
    return [f_block, BodyLeaf(op_idx=op.idx, phase="reduce_close")]


def _phase_dims_activation_reduce(op: ParsedOp) -> dict[str, tuple[str, ...]]:
    """Return the dims each activation_reduce phase touches.

    reduce_step runs inside F; reduce_close runs outside F.
    """
    p_dim = op.axis_map["P"]
    f_dim = op.axis_map["F"]
    return {"reduce_step": (p_dim, f_dim), "reduce_close": (p_dim,)}


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


def _canonical_key(node: "LoopNode | BodyLeaf") -> tuple:
    """Recursive structural key for a node.

    Two nodes (and their subtrees) produce equal keys iff they have the
    same tree shape, dim_ids, trip counts, roles, reduce_ops, and leaf
    op_idx / phase tags.
    """
    if isinstance(node, BodyLeaf):
        return ("leaf", node.op_idx, node.phase)
    return (
        "node",
        node.dim_id,
        node.trip_count,
        node.role.value,
        node.reduce_op,
        tuple(_canonical_key(c) for c in node.children),
    )


def hash_forest(forest: LoopForest) -> int:
    """Return a deterministic structural hash of ``forest``.

    Used by the ``tune`` stage's random-draw loop to break cycles
    caused by self-inverse rewrites (e.g. ``ReorderLoops`` applied
    twice restores the prior state).

    Covers only the forest — current structural rewrites leave
    ``op_graph`` untouched. Once graph rewrites land, extend the hash
    to include ``op_graph``.
    """
    return hash(tuple(_canonical_key(e) for e in forest))


def _resolve_node(forest: LoopForest, path: tuple[int, ...]) -> "LoopNode | BodyLeaf | None":
    """Walk ``path`` from the forest root; return the node at that position.

    Returns ``None`` when the path is invalid — empty, out of range, or
    traversing through a ``BodyLeaf``.
    """
    if not path:
        return None
    siblings: list[LoopNode | BodyLeaf] = list(forest)
    node: LoopNode | BodyLeaf | None = None
    for idx in path:
        if idx < 0 or idx >= len(siblings):
            return None
        node = siblings[idx]
        if isinstance(node, BodyLeaf):
            """Consumed the terminal leaf; further path indices are
            invalid."""
            siblings = []
        else:
            siblings = node.children
    return node
