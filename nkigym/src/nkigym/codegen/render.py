"""``render``: lower an :class:`OpGraph` to NKI source via the forest walker.

The walker traverses a :class:`LoopForest` and emits NKI source for
every node. By default, a canonical forest is built where each op gets
its own independent loop nest. SBUF intermediates are hoisted to
function top as full-extent ``(p_tile, num_p_tiles, num_f_tiles *
f_tile)`` allocations. HBM lives on kernel inputs (consumed by
``NKILoad``) and the kernel's return tensor (written by ``NKIStore``).
"""

from collections.abc import Callable

from nkigym.codegen.dep_graph import DepGraph
from nkigym.codegen.graph import OpGraph, Tensor
from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode, build_canonical_forest


def required_tiles(tensor: Tensor, dim_id: str, op_graph: OpGraph, forest: LoopForest) -> int:
    """Return the minimum tile count along ``dim_id`` that ``tensor`` must hold.

    Derived by walking the forest: find the LCA of ``tensor``'s producer
    and all consumers, then take
    ``num_tiles(dim_id) / product_of_dim_id_trips_above_lca``. For
    cross-loopnest tensors (LCA is the forest root) the product is 1
    and ``required_tiles`` equals ``num_tiles(dim_id)``. For fully
    intra-loopnest tensors (LCA below all ``dim_id``-iterating
    ancestors) the product equals ``num_tiles(dim_id)`` and
    ``required_tiles`` is ``1``.

    Parameter and return tensors — which live in HBM and are not
    tile-decomposed — return ``op_graph.dims[dim_id].num_tiles``
    unchanged.

    Raises:
        ValueError: Product of ancestor trip counts does not divide
            ``num_tiles(dim_id)`` (unsupported forest shape).
    """
    num_t = op_graph.dims[dim_id].num_tiles
    if tensor.origin in ("param", "return"):
        return num_t
    paths = _find_access_paths(tensor.name, op_graph, forest)
    if not paths:
        return num_t
    lca = _lowest_common_ancestor(paths)
    prod = 1
    for node in lca:
        if isinstance(node, LoopNode) and node.dim_id == dim_id:
            prod *= node.trip_count
    if num_t % prod != 0:
        raise ValueError(
            f"Tensor {tensor.name!r} dim {dim_id!r}: ancestor trip product {prod} " f"does not divide num_tiles {num_t}"
        )
    return num_t // prod


def _find_access_paths(tensor_name: str, op_graph: OpGraph, forest: LoopForest) -> list[list[LoopNode | BodyLeaf]]:
    """Return root-to-leaf node paths for every BodyLeaf that reads or writes ``tensor_name``.

    Each path is a list of ancestor nodes ending in the ``BodyLeaf``.
    The producer (unique by SSA) + all consumers are collected via
    ``op_graph.dep``.
    """
    producer_idx = op_graph.dep.producer.get(tensor_name)
    consumer_idxs = op_graph.dep.consumers.get(tensor_name, ())
    targets = set(consumer_idxs)
    if producer_idx is not None:
        targets.add(producer_idx)
    paths: list[list[LoopNode | BodyLeaf]] = []

    def walk(node: LoopNode | BodyLeaf, stack: list[LoopNode | BodyLeaf]) -> None:
        stack.append(node)
        if isinstance(node, BodyLeaf):
            if node.op_idx in targets:
                paths.append(list(stack))
        else:
            for child in node.children:
                walk(child, stack)
        stack.pop()

    for root in forest:
        walk(root, [])
    return paths


def _lowest_common_ancestor(paths: list[list[LoopNode | BodyLeaf]]) -> list[LoopNode | BodyLeaf]:
    """Return the longest common prefix of root-to-leaf paths."""
    if not paths:
        return []
    common = paths[0]
    for p in paths[1:]:
        new_len = 0
        for a, b in zip(common, p):
            if a is b:
                new_len += 1
            else:
                break
        common = common[:new_len]
    return common


def _sbuf_name(tensor_name: str) -> str:
    """Return the SBUF buffer name for a tensor.

    Strips a trailing ``_sbuf`` from the tensor name before prepending
    ``sbuf_`` so that user-supplied names like ``lhs_sbuf`` don't land
    as ``sbuf_lhs_sbuf`` in the emitted kernel.
    """
    stem = tensor_name[: -len("_sbuf")] if tensor_name.endswith("_sbuf") else tensor_name
    return f"sbuf_{stem}"


def _hbm_name(tensor_name: str) -> str:
    """Return the HBM buffer name for a tensor (dual of ``_sbuf_name``)."""
    stem = tensor_name[: -len("_sbuf")] if tensor_name.endswith("_sbuf") else tensor_name
    return f"hbm_{stem}"


class _Writer:
    """Line-based writer with indentation tracking."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._depth = 0

    def indent(self) -> None:
        """Open a nested block — subsequent ``line`` calls indent one level deeper."""
        self._depth += 1

    def dedent(self) -> None:
        """Close a nested block."""
        self._depth -= 1

    def line(self, text: str = "") -> None:
        """Append a source line at the current indent."""
        self._lines.append(("    " * self._depth + text) if text else "")

    def getvalue(self) -> str:
        """Return the accumulated source with a trailing newline."""
        return "\n".join(self._lines) + "\n"


def render(op_graph: OpGraph, forest: LoopForest | None = None) -> str:
    """Render ``op_graph`` to NKI kernel source via the forest walker.

    When ``forest`` is ``None``, a canonical forest is built from
    ``op_graph`` — matches today's default behaviour. Callers with a
    transformed forest (e.g. after fusion rewrites) pass it explicitly.
    """
    if forest is None:
        forest = build_canonical_forest(op_graph)
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, op_graph)
    w.indent()
    _emit_param_asserts(w, op_graph)
    _emit_hbm_output(w, op_graph)
    _emit_sbuf_allocations(w, op_graph, forest)
    render_forest(w, op_graph, forest)
    w.line(f"return {_hbm_name(op_graph.return_name)}")
    w.dedent()
    return w.getvalue()


def render_annotated(op_graph: OpGraph, forest: LoopForest | None = None) -> str:
    """Render like :func:`render`, but with forest-node comments inline.

    Every emitted ``for`` header is preceded by a ``# LoopNode(...)``
    comment, and every body dispatch is preceded by a
    ``# BodyLeaf(...)`` comment. Comments are standard Python lines, so
    the kernel runs unchanged.

    Shares the main render pipeline — only the forest walker is
    replaced with an annotating variant. All other emit helpers (SBUF
    allocation, signature, asserts, body emitters) are reused verbatim.
    """
    if forest is None:
        forest = build_canonical_forest(op_graph)
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, op_graph)
    w.indent()
    _emit_param_asserts(w, op_graph)
    _emit_hbm_output(w, op_graph)
    _emit_sbuf_allocations(w, op_graph, forest)
    _render_forest_annotated(w, op_graph, forest)
    w.line(f"return {_hbm_name(op_graph.return_name)}")
    w.dedent()
    return w.getvalue()


def _render_forest_annotated(w: _Writer, op_graph: OpGraph, forest: LoopForest) -> None:
    """Walk ``forest`` emitting source with per-node comments above each emission."""
    path_names: dict[str, list[str]] = {}
    path_trips: dict[str, list[int]] = {}
    for idx, entry in enumerate(forest):
        _emit_node_annotated(w, op_graph, entry, path_names, path_trips, forest, path=(idx,))


def _emit_node_annotated(
    w: _Writer,
    op_graph: OpGraph,
    node: LoopNode | BodyLeaf,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    forest: LoopForest,
    path: tuple[int, ...],
) -> None:
    """Like :func:`_emit_node` but emits `# LoopNode(...)` / `# BodyLeaf(...)` above each emission.

    Pipelined loops (``pipeline_depth > 1``) fall back to the un-annotated
    :func:`_emit_pipelined_loop` — the annotator is a read-only helper for
    the walkthrough doc and doesn't duplicate the prologue/body/epilogue
    machinery.
    """
    if isinstance(node, BodyLeaf):
        op = op_graph.ops[node.op_idx]
        emitter = _BODY_EMITTERS.get((op.op_cls.__name__, node.phase))
        if emitter is None:
            raise ValueError(f"No body emitter registered for ({op.op_cls.__name__!r}, {node.phase!r})")
        w.line(f'# BodyLeaf(op_idx={node.op_idx}, phase="{node.phase}")  path={path}')
        emitter(w, op_graph, op, path_names, path_trips, forest)
        return
    if node.pipeline_depth > 1:
        _emit_pipelined_loop(w, op_graph, node, path_names, path_trips, forest)
        return
    existing = path_names.setdefault(node.dim_id, [])
    loop_var = node.name if node.name is not None else f"i_{node.dim_id}_{len(existing)}"
    w.line(
        f'# LoopNode(dim_id="{node.dim_id}", trip={node.trip_count}, '
        f'role={node.role.name}, name="{node.name}")  path={path}'
    )
    w.line(f"for {loop_var} in range({node.trip_count}):")
    w.indent()
    existing.append(loop_var)
    path_trips.setdefault(node.dim_id, []).append(node.trip_count)
    for i, child in enumerate(node.children):
        _emit_node_annotated(w, op_graph, child, path_names, path_trips, forest, path=path + (i,))
    path_trips[node.dim_id].pop()
    existing.pop()
    w.dedent()


def _emit_imports(w: _Writer) -> None:
    """Emit the standard import header."""
    w.line("import nki")
    w.line("import nki.isa as nisa")
    w.line("import nki.language as nl")
    w.line()
    w.line()


def _emit_signature(w: _Writer, op_graph: OpGraph) -> None:
    """Emit ``@nki.jit`` + ``def <func>(<params>):``."""
    w.line("@nki.jit")
    params = ", ".join(op_graph.param_names)
    w.line(f"def {op_graph.func_name}({params}):")


def _emit_param_asserts(w: _Writer, op_graph: OpGraph) -> None:
    """Emit ``assert <param>.shape == (...)`` for every kernel input."""
    for name in op_graph.param_names:
        shape = op_graph.tensors[name].shape
        w.line(f"assert {name}.shape == {shape}")


def _emit_hbm_output(w: _Writer, op_graph: OpGraph) -> None:
    """Allocate the HBM output tensor (``hbm_<return>``)."""
    ret = op_graph.tensors[op_graph.return_name]
    w.line(f"{_hbm_name(ret.name)} = nl.ndarray({ret.shape}, dtype=nl.{ret.dtype}, buffer=nl.shared_hbm)")


def _emit_sbuf_allocations(w: _Writer, op_graph: OpGraph, forest: LoopForest) -> None:
    """Allocate one SBUF buffer per intermediate, then per op-local buffer.

    Kernel inputs live in HBM (consumed by ``NKILoad``) and the return
    tensor lives in HBM (written by ``NKIStore``). The store emitter
    reads from its data-operand's SBUF buffer directly, so the return
    has no SBUF mirror and is skipped here. Op-local buffers come after
    cross-nest tensors, in op-index order, sized at single-iteration
    scope (``(tile_size, 1, free_extent)``).
    """
    for name, tensor in op_graph.tensors.items():
        if tensor.origin in ("param", "return"):
            continue
        shape = _sbuf_shape(tensor, op_graph, forest)
        w.line(f"{_sbuf_name(name)} = nl.ndarray({shape}, dtype=nl.{tensor.dtype}, buffer=nl.sbuf)")
    for op in op_graph.ops:
        for buf in op.op_local_buffers.values():
            nl_buffer = "nl.sbuf" if buf.location == "sbuf" else "nl.psum"
            w.line(f"{buf.emitted_name} = nl.ndarray({buf.shape}, dtype=nl.{buf.dtype}, buffer={nl_buffer})")
    w.line()


def _sbuf_shape(tensor: Tensor, op_graph: OpGraph, forest: LoopForest) -> tuple[int, int, int]:
    """Compute 3D SBUF shape ``(p_tile, total_slots_P, num_f_tiles * f_tile)``.

    ``total_slots_P = required_tiles(P) * buffer_degree[P]``. Free axis
    still spans the full tile count for now — free-axis multi-buffer is
    out of scope.

    1D tensors collapse the free axis to a single element.
    """
    if not tensor.dim_ids:
        raise ValueError(f"Tensor {tensor.name!r} has no dims")
    p_axis = tensor.dim_ids[0]
    p_info = op_graph.dims[p_axis]
    p_required = required_tiles(tensor, p_axis, op_graph, forest)
    p_total = p_required * tensor.buffer_degree[p_axis]
    if len(tensor.dim_ids) == 1:
        return (p_info.tile_size, p_total, 1)
    f_axis = tensor.dim_ids[1]
    f_info = op_graph.dims[f_axis]
    return (p_info.tile_size, p_total, f_info.num_tiles * f_info.tile_size)


def _slot_expr(
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    dim_id: str,
    total_slots: int,
    stage_offset: int = 0,
) -> str:
    """Return the slot expression for ``dim_id``.

    For dim ``d`` with ``k`` same-dim ancestors on the current path and
    loop variable names ``path_names[d] = [n_0, n_1, ..., n_{k-1}]``
    (outermost→innermost), the raw slot is ``sum_{idx} n_idx *
    prod_of_tail_trips``. The final expression is
    ``(raw_slot) % total_slots``, with two simplifications:

    * ``total_slots == 1`` — slot is literal ``"0"``.
    * ``total_slots == product_of_ancestor_trips`` (raw slot never
      exceeds ``total_slots``) — modulo is identity; emit the raw slot.

    ``stage_offset`` adds an integer offset to the innermost ancestor
    (used by the software-pipelined body emission in Task 8). Default 0.

    Uses each ancestor's persisted ``LoopNode.name`` so loop identity
    survives structural rewrites — post-swap, the same loop prints the
    same variable name regardless of its tree position.

    Raises:
        ValueError: ``dim_id`` has no open ancestor loops on the path.
    """
    names = path_names.get(dim_id, [])
    k = len(names)
    if k == 0:
        raise ValueError(f"No open LoopNode on path for dim {dim_id!r}")
    if total_slots == 1:
        return "0"
    trips = path_trips[dim_id]
    raw_trip_product = 1
    for t in trips:
        raw_trip_product *= t
    terms: list[str] = []
    for idx in range(k):
        tail_prod = 1
        for t in trips[idx + 1 :]:
            tail_prod *= t
        innermost = idx == k - 1
        if innermost and stage_offset != 0:
            sign = "+" if stage_offset > 0 else "-"
            token = f"({names[idx]} {sign} {abs(stage_offset)})"
        else:
            token = names[idx]
        if tail_prod == 1:
            terms.append(token)
        else:
            terms.append(f"{token} * {tail_prod}")
    raw = " + ".join(terms)
    if total_slots == raw_trip_product and stage_offset == 0:
        return raw
    return f"({raw}) % {total_slots}"


def _tensor_total_slots(tensor: Tensor, dim_id: str, op_graph: OpGraph, forest: LoopForest) -> int:
    """Per-dim total slot count for a tensor: required_tiles * buffer_degree."""
    return required_tiles(tensor, dim_id, op_graph, forest) * tensor.buffer_degree[dim_id]


def _sbuf_tile_slice(
    name: str,
    dim_ids: tuple[str, ...],
    p_tile: int,
    f_tile: int,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    total_slots_p: int,
    total_slots_f: int,
    stage_offset_p: int = 0,
    stage_offset_f: int = 0,
) -> str:
    """Return the SBUF ``[p_tile, p_slot, f_range]`` slice expression."""
    p_axis = dim_ids[0]
    p_slot = _slot_expr(path_names, path_trips, p_axis, total_slots_p, stage_offset_p)
    if len(dim_ids) == 1:
        return f"{name}[0:{p_tile}, {p_slot}, 0:1]"
    f_axis = dim_ids[1]
    f_slot_inner = _slot_expr(path_names, path_trips, f_axis, total_slots_f, stage_offset_f)
    f_slot = f"({f_slot_inner})"
    return f"{name}[0:{p_tile}, {p_slot}, {f_slot} * {f_tile} : {f_slot} * {f_tile} + {f_tile}]"


def _hbm_tile_slice(
    name: str,
    dim_ids: tuple[str, ...],
    p_tile: int,
    f_tile: int,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    total_slots_p: int,
    total_slots_f: int,
    stage_offset_p: int = 0,
    stage_offset_f: int = 0,
) -> str:
    """Return the HBM ``[p_range, f_range]`` slice expression."""
    p_axis = dim_ids[0]
    p_slot_inner = _slot_expr(path_names, path_trips, p_axis, total_slots_p, stage_offset_p)
    p_slot = f"({p_slot_inner})"
    if len(dim_ids) == 1:
        return f"{name}[{p_slot} * {p_tile} : {p_slot} * {p_tile} + {p_tile}]"
    f_axis = dim_ids[1]
    f_slot_inner = _slot_expr(path_names, path_trips, f_axis, total_slots_f, stage_offset_f)
    f_slot = f"({f_slot_inner})"
    return (
        f"{name}[{p_slot} * {p_tile} : {p_slot} * {p_tile} + {p_tile}, "
        f"{f_slot} * {f_tile} : {f_slot} * {f_tile} + {f_tile}]"
    )


def _swapped_dst_tile_slice(
    dst_name: str,
    src_p_axis: str,
    src_f_axis: str,
    tile: int,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    total_slots_p: int,
    total_slots_f: int,
    stage_offset_dst_p: int = 0,
    stage_offset_dst_f: int = 0,
) -> str:
    """SBUF slice for a transpose's dst tensor (swapped axes).

    The dst's partition slot uses the source's free-axis ordinals; the
    dst's free slot uses the source's partition-axis ordinals. Transpose
    ops enforce square tiles (p_tile == f_tile), so a single ``tile``
    parameter suffices. ``total_slots_p`` is the slot count for the
    dst's partition axis (= src's free axis); ``total_slots_f`` is for
    the dst's free axis (= src's partition axis).

    ``stage_offset_dst_p`` applies to the innermost ancestor of
    ``src_f_axis`` (used by the dst P slot); ``stage_offset_dst_f``
    applies to the innermost ancestor of ``src_p_axis`` (used by the
    dst F slot).
    """
    p_slot = _slot_expr(path_names, path_trips, src_f_axis, total_slots_p, stage_offset_dst_p)
    f_slot_inner = _slot_expr(path_names, path_trips, src_p_axis, total_slots_f, stage_offset_dst_f)
    f_slot = f"({f_slot_inner})"
    return f"{_sbuf_name(dst_name)}[0:{tile}, {p_slot}, " f"{f_slot} * {tile} : {f_slot} * {tile} + {tile}]"


_REDUCE_IDENTITY: dict[str, float] = {"add": 0.0, "max": float("-inf")}
_REDUCE_MERGE_OP: dict[str, str] = {"add": "nl.add", "max": "nl.maximum"}


_BODY_EMITTERS: dict[tuple[str, str], Callable] = {}
"""Per-``(op_kind, phase)`` body emitter registry.

A body emitter receives ``(writer, op_graph, parsed_op, path_names,
path_trips, forest)`` and emits that phase's source lines without any
loop headers — the walker is responsible for opening and closing the
loops that frame the body. ``forest`` is threaded so emitters can call
:func:`_tensor_total_slots` to derive each operand's per-dim
``total_slots`` for slot expressions. Single-phase ops register under
phase ``"main"``.
"""


def _register_body(op_kind: str, phase: str = "main"):
    """Decorator: register a body emitter for ``(op_kind, phase)``."""

    def wrap(fn: Callable) -> Callable:
        _BODY_EMITTERS[(op_kind, phase)] = fn
        return fn

    return wrap


def render_forest(w: _Writer, op_graph: OpGraph, forest: LoopForest) -> None:
    """Walk ``forest`` and emit NKI source for every node.

    ``path_names[d]`` is the list of same-dim ancestor loop variable
    names (outermost->innermost) open above the current position;
    ``path_trips[d]`` carries their trip counts in the same order.
    Body emitters use both to build slot expressions via
    :func:`_slot_expr`, and consult ``forest`` to derive each tensor's
    ``total_slots`` per dim.
    """
    path_names: dict[str, list[str]] = {}
    path_trips: dict[str, list[int]] = {}
    for entry in forest:
        _emit_node(w, op_graph, entry, path_names, path_trips, forest)


def assign_stages(loop_node: LoopNode, dep: DepGraph) -> dict[tuple[int, str], int]:
    """Return a stage index per ``(op_idx, phase)`` leaf in ``loop_node``'s subtree.

    Walks the subtree in source (DFS, children-order) order; each leaf's
    stage is one more than the max stage of any upstream leaf (earlier
    in the walk) whose writes it reads. Leaves that read nothing
    produced in the subtree get stage 0.

    Args:
        loop_node: The ``LoopNode`` whose subtree's stages are assigned.
        dep: Op-level dep graph.

    Returns:
        Dict keyed by ``(op_idx, phase)`` mapping to the leaf's stage.
    """
    leaves = _collect_body_leaves(loop_node)
    stage: dict[tuple[int, str], int] = {}
    writes_in_subtree: dict[str, tuple[int, str]] = {}
    for leaf in leaves:
        reads = dep.reads.get(leaf.op_idx, frozenset())
        upstream = [stage[writes_in_subtree[t]] for t in reads if t in writes_in_subtree]
        stage[(leaf.op_idx, leaf.phase)] = (max(upstream) + 1) if upstream else 0
        for t in dep.writes.get(leaf.op_idx, frozenset()):
            writes_in_subtree[t] = (leaf.op_idx, leaf.phase)
    return stage


def _collect_body_leaves(node: LoopNode | BodyLeaf) -> list[BodyLeaf]:
    """Gather every ``BodyLeaf`` under ``node`` in tree (DFS) order."""
    acc: list[BodyLeaf] = []

    def walk(n: LoopNode | BodyLeaf) -> None:
        if isinstance(n, BodyLeaf):
            acc.append(n)
        else:
            for c in n.children:
                walk(c)

    walk(node)
    return acc


def _emit_node(
    w: _Writer,
    op_graph: OpGraph,
    node: LoopNode | BodyLeaf,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    forest: LoopForest,
) -> None:
    """Emit one forest node (recursive for ``LoopNode``, delegating for ``BodyLeaf``).

    For ``LoopNode``, dispatches to the vanilla or pipelined emitter
    based on ``pipeline_depth``. ``BodyLeaf`` delegates to the registry.
    """
    if isinstance(node, BodyLeaf):
        op = op_graph.ops[node.op_idx]
        emitter = _BODY_EMITTERS.get((op.op_cls.__name__, node.phase))
        if emitter is None:
            raise ValueError(f"No body emitter registered for ({op.op_cls.__name__!r}, {node.phase!r})")
        emitter(w, op_graph, op, path_names, path_trips, forest)
        return
    if node.pipeline_depth <= 1:
        _emit_vanilla_loop(w, op_graph, node, path_names, path_trips, forest)
    else:
        _emit_pipelined_loop(w, op_graph, node, path_names, path_trips, forest)


def _emit_vanilla_loop(
    w: _Writer,
    op_graph: OpGraph,
    node: LoopNode,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    forest: LoopForest,
) -> None:
    """Emit the un-pipelined ``for`` loop for ``pipeline_depth == 1``.

    Uses each ``LoopNode.name`` as the emitted loop variable so loop
    identity survives structural rewrites (reorder, fusion). Falls back
    to a position-derived name (``i_<dim_id>_<ordinal>``) when ``name``
    is unset — used by hand-constructed test forests.
    """
    existing = path_names.setdefault(node.dim_id, [])
    loop_var = node.name if node.name is not None else f"i_{node.dim_id}_{len(existing)}"
    w.line(f"for {loop_var} in range({node.trip_count}):")
    w.indent()
    existing.append(loop_var)
    path_trips.setdefault(node.dim_id, []).append(node.trip_count)
    for child in node.children:
        _emit_node(w, op_graph, child, path_names, path_trips, forest)
    path_trips[node.dim_id].pop()
    existing.pop()
    w.dedent()


def _emit_pipelined_loop(
    w: _Writer,
    op_graph: OpGraph,
    node: LoopNode,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    forest: LoopForest,
) -> None:
    """Emit prologue (``D-1`` iters) + pipelined body + epilogue (``D-1`` iters).

    Let ``D = node.pipeline_depth`` and ``N = node.trip_count``; each
    leaf in the subtree carries a stage ``s ∈ [0, max_stage]`` from
    :func:`assign_stages`. Semantics for each of the three phases:

    * **Prologue** — ``D-1`` unrolled iters, ``i_pro ∈ [0, D-2]``. At
      iter ``i_pro``, every leaf at stage ``s ≤ i_pro`` fires with the
      pipelined dim's innermost ancestor bound to the integer literal
      ``i_pro - s`` (element index the stage is processing). Leaves
      with ``s > i_pro`` do not fire.
    * **Pipelined body** — ``for loop_var in range(D-1, N):``. Every
      leaf fires once per iter with ``stage_offset = -s`` threaded to
      the emitter so the innermost ``loop_var`` is substituted by
      ``(loop_var - s)``.
    * **Epilogue** — ``D-1`` unrolled iters, ``i_epi ∈ [0, D-2]``. At
      iter ``i_epi``, every leaf at stage ``s > i_epi`` fires with the
      pipelined dim's innermost ancestor bound to the integer literal
      ``N + i_epi - s``. Leaves with ``s ≤ i_epi`` do not fire.

    Total firings per leaf across all three phases equal ``N``, matching
    the un-pipelined case.

    Raises:
        ValueError: ``pipeline_depth`` > ``max_stage + 1`` (no chain to
            pipeline that deep), ``pipeline_depth`` > ``trip_count``
            (not enough iterations for the skew), or the subtree has no
            body leaves at all.
    """
    stages = assign_stages(node, op_graph.dep)
    if not stages:
        raise ValueError(f"pipeline_depth>1 on LoopNode(dim_id={node.dim_id!r}) with no body leaves")
    max_stage = max(stages.values())
    depth = node.pipeline_depth
    trip = node.trip_count
    if depth > max_stage + 1:
        raise ValueError(f"pipeline_depth {depth} exceeds chain length {max_stage + 1} in subtree")
    if trip < depth:
        raise ValueError(f"pipeline_depth {depth} exceeds trip_count {trip}")

    existing = path_names.setdefault(node.dim_id, [])
    loop_var = node.name if node.name is not None else f"i_{node.dim_id}_{len(existing)}"
    path_trips.setdefault(node.dim_id, []).append(trip)

    for i_pro in range(depth - 1):
        for child in node.children:
            _emit_pipelined_leaf(
                w,
                op_graph,
                child,
                path_names,
                path_trips,
                forest,
                stages,
                node.dim_id,
                constant_loop_var=i_pro,
                fire_if_stage_le=i_pro,
                fire_if_stage_gt=None,
            )

    w.line(f"for {loop_var} in range({depth - 1}, {trip}):")
    w.indent()
    existing.append(loop_var)
    for child in node.children:
        _emit_pipelined_leaf(
            w,
            op_graph,
            child,
            path_names,
            path_trips,
            forest,
            stages,
            node.dim_id,
            constant_loop_var=None,
            fire_if_stage_le=None,
            fire_if_stage_gt=None,
        )
    existing.pop()
    w.dedent()

    for i_epi in range(depth - 1):
        absolute = trip + i_epi
        for child in node.children:
            _emit_pipelined_leaf(
                w,
                op_graph,
                child,
                path_names,
                path_trips,
                forest,
                stages,
                node.dim_id,
                constant_loop_var=absolute,
                fire_if_stage_le=None,
                fire_if_stage_gt=i_epi,
            )

    path_trips[node.dim_id].pop()


def _subtree_has_firing_leaf(
    node: LoopNode | BodyLeaf,
    stages: dict[tuple[int, str], int],
    fire_if_stage_le: int | None,
    fire_if_stage_gt: int | None,
) -> bool:
    """Return ``True`` when at least one BodyLeaf under ``node`` passes the stage gate.

    Used to suppress empty ``for`` headers during pipelined prologue /
    epilogue emission: if every leaf in an inner subtree is gated out,
    we skip the wrapping loop entirely rather than emit a body-less
    Python ``for`` block.
    """
    if isinstance(node, BodyLeaf):
        stage = stages.get((node.op_idx, node.phase))
        if stage is None:
            return False
        if fire_if_stage_le is not None and stage > fire_if_stage_le:
            return False
        if fire_if_stage_gt is not None and stage <= fire_if_stage_gt:
            return False
        return True
    return any(_subtree_has_firing_leaf(c, stages, fire_if_stage_le, fire_if_stage_gt) for c in node.children)


def _emit_pipelined_leaf(
    w: _Writer,
    op_graph: OpGraph,
    child: LoopNode | BodyLeaf,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    forest: LoopForest,
    stages: dict[tuple[int, str], int],
    dim_id: str,
    constant_loop_var: int | None,
    fire_if_stage_le: int | None,
    fire_if_stage_gt: int | None,
) -> None:
    """Emit one node under a pipelined loop.

    LoopNode children (any ``trip_count``) descend transparently. The
    pipelined skew lives at BodyLeaf granularity: each leaf's stage —
    assigned across the whole subtree by :func:`assign_stages` — drives
    the stage-gate and iteration offset, regardless of how many inner
    loops wrap it. Trivial ``trip_count == 1`` LoopNodes skip the
    ``for`` header and bind their loop variable to the literal ``"0"``;
    non-trivial LoopNodes emit a normal ``for`` header with their
    loop variable name pushed onto ``path_names``. Either way, children
    are emitted under the same stage-gate + constant-bind state.
    Non-trivial descent is required because ``FuseLoops`` produces
    canonical 2N forms (``Loop(d,N) -> Loop(d,1) -> body``) where the
    pipelined dim's trivial wrapper hides deeper non-pipelined loops
    (e.g. load's inner free-axis loop) that still need to run once per
    pipelined iter. Before descending we call
    :func:`_subtree_has_firing_leaf` to skip subtrees whose leaves are
    all gated out — emitting an empty ``for`` block would produce
    invalid Python.

    BodyLeaf emission has two modes:

    * ``constant_loop_var`` is an ``int`` — prologue or epilogue iter.
      Push ``str(constant_loop_var - stage)`` onto
      ``path_names[dim_id]`` as a literal ancestor name, call the
      emitter with ``pipeline_dim=None, stage_offset=0`` (literal
      already accounts for the stage), then pop.
    * ``constant_loop_var`` is ``None`` — pipelined body iter. The
      pipelined ``loop_var`` is already on ``path_names``; call the
      emitter with ``pipeline_dim=dim_id, stage_offset=-stage`` so the
      innermost ``dim_id`` ancestor is rewritten as ``(loop_var -
      stage)``.

    Stage-gating:

    * ``fire_if_stage_le`` not ``None`` — fire only if ``stage ≤ this``.
    * ``fire_if_stage_gt`` not ``None`` — fire only if ``stage > this``.
    * Both ``None`` — fire unconditionally (body path).
    """
    if isinstance(child, LoopNode):
        if not _subtree_has_firing_leaf(child, stages, fire_if_stage_le, fire_if_stage_gt):
            return
        existing = path_names.setdefault(child.dim_id, [])
        if child.trip_count == 1:
            """Trivial trip=1 loop: skip for-header emission, bind var to
            literal ``"0"`` so inner slot expressions see the single
            valid iteration index."""
            existing.append("0")
            path_trips.setdefault(child.dim_id, []).append(1)
            try:
                for grandchild in child.children:
                    _emit_pipelined_leaf(
                        w,
                        op_graph,
                        grandchild,
                        path_names,
                        path_trips,
                        forest,
                        stages,
                        dim_id,
                        constant_loop_var,
                        fire_if_stage_le,
                        fire_if_stage_gt,
                    )
            finally:
                path_trips[child.dim_id].pop()
                existing.pop()
            return
        """Non-trivial inner loop: emit the ``for`` header as vanilla;
        inner body leaves still honor the outer pipelined stage-gate."""
        loop_var = child.name if child.name is not None else f"i_{child.dim_id}_{len(existing)}"
        w.line(f"for {loop_var} in range({child.trip_count}):")
        w.indent()
        existing.append(loop_var)
        path_trips.setdefault(child.dim_id, []).append(child.trip_count)
        try:
            for grandchild in child.children:
                _emit_pipelined_leaf(
                    w,
                    op_graph,
                    grandchild,
                    path_names,
                    path_trips,
                    forest,
                    stages,
                    dim_id,
                    constant_loop_var,
                    fire_if_stage_le,
                    fire_if_stage_gt,
                )
        finally:
            path_trips[child.dim_id].pop()
            existing.pop()
            w.dedent()
        return

    stage = stages[(child.op_idx, child.phase)]
    if fire_if_stage_le is not None and stage > fire_if_stage_le:
        return
    if fire_if_stage_gt is not None and stage <= fire_if_stage_gt:
        return
    op = op_graph.ops[child.op_idx]
    emitter = _BODY_EMITTERS[(op.op_cls.__name__, child.phase)]

    if constant_loop_var is not None:
        literal_value = constant_loop_var - stage
        existing = path_names.setdefault(dim_id, [])
        existing.append(str(literal_value))
        try:
            emitter(w, op_graph, op, path_names, path_trips, forest)
        finally:
            existing.pop()
    else:
        emitter(w, op_graph, op, path_names, path_trips, forest, pipeline_dim=dim_id, stage_offset=-stage)


@_register_body("NKILoad", "main")
def _body_load(w, op_graph, op, path_names, path_trips, forest, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.dma_copy`` at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src_tensor = op_graph.tensors[src_name]
    dst_tensor = op_graph.tensors[dst_name]
    p_axis = src_tensor.dim_ids[0]
    f_axis = src_tensor.dim_ids[1] if len(src_tensor.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    dst_p_slots = _tensor_total_slots(dst_tensor, dst_tensor.dim_ids[0], op_graph, forest)
    dst_f_slots = (
        _tensor_total_slots(dst_tensor, dst_tensor.dim_ids[1], op_graph, forest) if len(dst_tensor.dim_ids) > 1 else 1
    )
    src_p_slots = op_graph.dims[src_tensor.dim_ids[0]].num_tiles
    src_f_slots = op_graph.dims[src_tensor.dim_ids[1]].num_tiles if len(src_tensor.dim_ids) > 1 else 1
    off_p = stage_offset if p_axis == pipeline_dim else 0
    off_f = stage_offset if f_axis is not None and f_axis == pipeline_dim else 0
    dst_expr = _sbuf_tile_slice(
        _sbuf_name(dst_name),
        dst_tensor.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        dst_p_slots,
        dst_f_slots,
        off_p,
        off_f,
    )
    src_expr = _hbm_tile_slice(
        src_name, src_tensor.dim_ids, p_tile, f_tile, path_names, path_trips, src_p_slots, src_f_slots, off_p, off_f
    )
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")


@_register_body("NKIStore", "main")
def _body_store(w, op_graph, op, path_names, path_trips, forest, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.dma_copy`` SBUF→HBM at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src_tensor = op_graph.tensors[src_name]
    dst_tensor = op_graph.tensors[dst_name]
    p_axis = dst_tensor.dim_ids[0]
    f_axis = dst_tensor.dim_ids[1] if len(dst_tensor.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    dst_p_slots = op_graph.dims[dst_tensor.dim_ids[0]].num_tiles
    dst_f_slots = op_graph.dims[dst_tensor.dim_ids[1]].num_tiles if len(dst_tensor.dim_ids) > 1 else 1
    src_p_slots = _tensor_total_slots(src_tensor, src_tensor.dim_ids[0], op_graph, forest)
    src_f_slots = (
        _tensor_total_slots(src_tensor, src_tensor.dim_ids[1], op_graph, forest) if len(src_tensor.dim_ids) > 1 else 1
    )
    off_p = stage_offset if p_axis == pipeline_dim else 0
    off_f = stage_offset if f_axis is not None and f_axis == pipeline_dim else 0
    dst_expr = _hbm_tile_slice(
        _hbm_name(dst_name),
        dst_tensor.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        dst_p_slots,
        dst_f_slots,
        off_p,
        off_f,
    )
    src_expr = _sbuf_tile_slice(
        _sbuf_name(src_name),
        src_tensor.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        src_p_slots,
        src_f_slots,
        off_p,
        off_f,
    )
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")


@_register_body("NKIActivation", "main")
def _body_activation(w, op_graph, op, path_names, path_trips, forest, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.activation`` at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    dst = op_graph.tensors[dst_name]
    p_axis = src.dim_ids[0]
    f_axis = src.dim_ids[1] if len(src.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    act = op.op_kwargs["op"]
    scale = op.op_kwargs.get("scale", 1.0)
    bias = op.op_kwargs.get("bias", 0.0)
    dst_p_slots = _tensor_total_slots(dst, dst.dim_ids[0], op_graph, forest)
    dst_f_slots = _tensor_total_slots(dst, dst.dim_ids[1], op_graph, forest) if len(dst.dim_ids) > 1 else 1
    src_p_slots = _tensor_total_slots(src, src.dim_ids[0], op_graph, forest)
    src_f_slots = _tensor_total_slots(src, src.dim_ids[1], op_graph, forest) if len(src.dim_ids) > 1 else 1
    off_p = stage_offset if p_axis == pipeline_dim else 0
    off_f = stage_offset if f_axis is not None and f_axis == pipeline_dim else 0
    dst_expr = _sbuf_tile_slice(
        _sbuf_name(dst_name),
        dst.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        dst_p_slots,
        dst_f_slots,
        off_p,
        off_f,
    )
    src_expr = _sbuf_tile_slice(
        _sbuf_name(src_name),
        src.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        src_p_slots,
        src_f_slots,
        off_p,
        off_f,
    )
    w.line(f"nisa.activation(dst={dst_expr}, op=nl.{act}, data={src_expr}, scale={scale}, bias={bias})")


@_register_body("NKITensorScalar", "main")
def _body_tensor_scalar(w, op_graph, op, path_names, path_trips, forest, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.tensor_scalar`` at the innermost open-loop point."""
    data_name = op.operand_names["data"]
    op0_name = op.operand_names["operand0"]
    dst_name = op.output_names[0]
    data = op_graph.tensors[data_name]
    op0 = op_graph.tensors[op0_name]
    dst = op_graph.tensors[dst_name]
    p_axis = data.dim_ids[0]
    f_axis = data.dim_ids[1]
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size
    op_name = op.op_kwargs["op"]
    dst_p_slots = _tensor_total_slots(dst, dst.dim_ids[0], op_graph, forest)
    dst_f_slots = _tensor_total_slots(dst, dst.dim_ids[1], op_graph, forest) if len(dst.dim_ids) > 1 else 1
    data_p_slots = _tensor_total_slots(data, data.dim_ids[0], op_graph, forest)
    data_f_slots = _tensor_total_slots(data, data.dim_ids[1], op_graph, forest) if len(data.dim_ids) > 1 else 1
    op0_p_slots = _tensor_total_slots(op0, op0.dim_ids[0], op_graph, forest)
    op0_f_slots = _tensor_total_slots(op0, op0.dim_ids[1], op_graph, forest) if len(op0.dim_ids) > 1 else 1
    """Offset per-operand: stage_offset applies to the ancestors of the
    pipelined dim. Each operand's axes compare independently (op0 may
    be 1D and thus lack an f-axis)."""
    dst_off_p = stage_offset if dst.dim_ids[0] == pipeline_dim else 0
    dst_off_f = stage_offset if len(dst.dim_ids) > 1 and dst.dim_ids[1] == pipeline_dim else 0
    data_off_p = stage_offset if data.dim_ids[0] == pipeline_dim else 0
    data_off_f = stage_offset if len(data.dim_ids) > 1 and data.dim_ids[1] == pipeline_dim else 0
    op0_off_p = stage_offset if op0.dim_ids[0] == pipeline_dim else 0
    op0_off_f = stage_offset if len(op0.dim_ids) > 1 and op0.dim_ids[1] == pipeline_dim else 0
    dst_expr = _sbuf_tile_slice(
        _sbuf_name(dst_name),
        dst.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        dst_p_slots,
        dst_f_slots,
        dst_off_p,
        dst_off_f,
    )
    data_expr = _sbuf_tile_slice(
        _sbuf_name(data_name),
        data.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        data_p_slots,
        data_f_slots,
        data_off_p,
        data_off_f,
    )
    op0_expr = _sbuf_tile_slice(
        _sbuf_name(op0_name),
        op0.dim_ids,
        p_tile,
        1,
        path_names,
        path_trips,
        op0_p_slots,
        op0_f_slots,
        op0_off_p,
        op0_off_f,
    )
    w.line(f"nisa.tensor_scalar(dst={dst_expr}, data={data_expr}, op0=nl.{op_name}, operand0={op0_expr})")


@_register_body("NKITranspose", "main")
def _body_transpose(w, op_graph, op, path_names, path_trips, forest, pipeline_dim=None, stage_offset=0) -> None:
    """Emit PSUM alloc + ``nc_transpose`` + ``tensor_copy`` at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    dst = op_graph.tensors[dst_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = op_graph.dims[src_p_axis].tile_size
    f_tile = op_graph.dims[src_f_axis].tile_size
    w.line(f"psum_tile = nl.ndarray(({p_tile}, {f_tile}), dtype=nl.{dst.dtype}, buffer=nl.psum)")
    src_p_slots = _tensor_total_slots(src, src.dim_ids[0], op_graph, forest)
    src_f_slots = _tensor_total_slots(src, src.dim_ids[1], op_graph, forest)
    src_off_p = stage_offset if src_p_axis == pipeline_dim else 0
    src_off_f = stage_offset if src_f_axis == pipeline_dim else 0
    src_expr = _sbuf_tile_slice(
        _sbuf_name(src_name),
        src.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        src_p_slots,
        src_f_slots,
        src_off_p,
        src_off_f,
    )
    w.line(f"nisa.nc_transpose(psum_tile[0:{p_tile}, 0:{f_tile}], {src_expr})")
    dst_p_slots = _tensor_total_slots(dst, dst.dim_ids[0], op_graph, forest)
    dst_f_slots = _tensor_total_slots(dst, dst.dim_ids[1], op_graph, forest)
    """dst's P slot uses src_f_axis ancestors; dst's F slot uses src_p_axis ancestors."""
    dst_off_p = stage_offset if src_f_axis == pipeline_dim else 0
    dst_off_f = stage_offset if src_p_axis == pipeline_dim else 0
    dst_expr = _swapped_dst_tile_slice(
        dst_name, src_p_axis, src_f_axis, p_tile, path_names, path_trips, dst_p_slots, dst_f_slots, dst_off_p, dst_off_f
    )
    w.line(f"nisa.tensor_copy({dst_expr}, psum_tile[0:{p_tile}, 0:{f_tile}])")


@_register_body("NKIDMATranspose", "main")
def _body_dma_transpose(w, op_graph, op, path_names, path_trips, forest, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.dma_transpose`` at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    dst = op_graph.tensors[dst_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = op_graph.dims[src_p_axis].tile_size
    f_tile = op_graph.dims[src_f_axis].tile_size
    src_p_slots = _tensor_total_slots(src, src.dim_ids[0], op_graph, forest)
    src_f_slots = _tensor_total_slots(src, src.dim_ids[1], op_graph, forest)
    src_off_p = stage_offset if src_p_axis == pipeline_dim else 0
    src_off_f = stage_offset if src_f_axis == pipeline_dim else 0
    src_expr = _sbuf_tile_slice(
        _sbuf_name(src_name),
        src.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        src_p_slots,
        src_f_slots,
        src_off_p,
        src_off_f,
    )
    dst_p_slots = _tensor_total_slots(dst, dst.dim_ids[0], op_graph, forest)
    dst_f_slots = _tensor_total_slots(dst, dst.dim_ids[1], op_graph, forest)
    dst_off_p = stage_offset if src_f_axis == pipeline_dim else 0
    dst_off_f = stage_offset if src_p_axis == pipeline_dim else 0
    dst_expr = _swapped_dst_tile_slice(
        dst_name, src_p_axis, src_f_axis, p_tile, path_names, path_trips, dst_p_slots, dst_f_slots, dst_off_p, dst_off_f
    )
    w.line(f"nisa.dma_transpose({dst_expr}, {src_expr})")


@_register_body("NKIMatmul", "psum_init")
def _body_matmul_psum_init(w, op_graph, op, path_names, path_trips, forest, pipeline_dim=None, stage_offset=0) -> None:
    """Allocate + memset the PSUM accumulator once per (M, N) tile.

    PSUM lifetime spans the entire K loop. ``path_names`` / ``path_trips``
    / ``forest`` are unused — the alloc uses constant ``(p_tile_M,
    f_tile_N)`` shapes derived from the op's axis map. ``pipeline_dim``
    / ``stage_offset`` are unused because the alloc is index-free.
    """
    _ = path_names, path_trips, forest, pipeline_dim, stage_offset
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    p_tile_M = op_graph.dims[m_dim].tile_size
    f_tile_N = op_graph.dims[n_dim].tile_size
    w.line(f"psum_tile = nl.ndarray(({p_tile_M}, {f_tile_N}), dtype=nl.float32, buffer=nl.psum)")
    w.line(f"nisa.memset(psum_tile[0:{p_tile_M}, 0:{f_tile_N}], value=0.0)")


@_register_body("NKIMatmul", "compute")
def _body_matmul_compute(w, op_graph, op, path_names, path_trips, forest, pipeline_dim=None, stage_offset=0) -> None:
    """Emit one ``nisa.nc_matmul`` per K tile inside the K loop."""
    stat_name = op.operand_names["stationary"]
    mov_name = op.operand_names["moving"]
    stat = op_graph.tensors[stat_name]
    mov = op_graph.tensors[mov_name]
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    k_dim = op.axis_map["K"]
    p_tile_M = op_graph.dims[m_dim].tile_size
    f_tile_N = op_graph.dims[n_dim].tile_size
    p_tile_K = op_graph.dims[k_dim].tile_size
    stat_p_slots = _tensor_total_slots(stat, stat.dim_ids[0], op_graph, forest)
    stat_f_slots = _tensor_total_slots(stat, stat.dim_ids[1], op_graph, forest) if len(stat.dim_ids) > 1 else 1
    mov_p_slots = _tensor_total_slots(mov, mov.dim_ids[0], op_graph, forest)
    mov_f_slots = _tensor_total_slots(mov, mov.dim_ids[1], op_graph, forest) if len(mov.dim_ids) > 1 else 1
    stat_off_p = stage_offset if stat.dim_ids[0] == pipeline_dim else 0
    stat_off_f = stage_offset if len(stat.dim_ids) > 1 and stat.dim_ids[1] == pipeline_dim else 0
    mov_off_p = stage_offset if mov.dim_ids[0] == pipeline_dim else 0
    mov_off_f = stage_offset if len(mov.dim_ids) > 1 and mov.dim_ids[1] == pipeline_dim else 0
    stat_expr = _sbuf_tile_slice(
        _sbuf_name(stat_name),
        stat.dim_ids,
        p_tile_K,
        p_tile_M,
        path_names,
        path_trips,
        stat_p_slots,
        stat_f_slots,
        stat_off_p,
        stat_off_f,
    )
    mov_expr = _sbuf_tile_slice(
        _sbuf_name(mov_name),
        mov.dim_ids,
        p_tile_K,
        f_tile_N,
        path_names,
        path_trips,
        mov_p_slots,
        mov_f_slots,
        mov_off_p,
        mov_off_f,
    )
    w.line("nisa.nc_matmul(")
    w.indent()
    w.line(f"dst=psum_tile[0:{p_tile_M}, 0:{f_tile_N}],")
    w.line(f"stationary={stat_expr},")
    w.line(f"moving={mov_expr},")
    w.dedent()
    w.line(")")


@_register_body("NKIMatmul", "drain")
def _body_matmul_drain(w, op_graph, op, path_names, path_trips, forest, pipeline_dim=None, stage_offset=0) -> None:
    """Drain the PSUM accumulator into the output SBUF once the K loop closes."""
    out_name = op.output_names[0]
    out = op_graph.tensors[out_name]
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    p_tile_M = op_graph.dims[m_dim].tile_size
    f_tile_N = op_graph.dims[n_dim].tile_size
    out_p_slots = _tensor_total_slots(out, out.dim_ids[0], op_graph, forest)
    out_f_slots = _tensor_total_slots(out, out.dim_ids[1], op_graph, forest) if len(out.dim_ids) > 1 else 1
    out_off_p = stage_offset if out.dim_ids[0] == pipeline_dim else 0
    out_off_f = stage_offset if len(out.dim_ids) > 1 and out.dim_ids[1] == pipeline_dim else 0
    out_expr = _sbuf_tile_slice(
        _sbuf_name(out_name),
        out.dim_ids,
        p_tile_M,
        f_tile_N,
        path_names,
        path_trips,
        out_p_slots,
        out_f_slots,
        out_off_p,
        out_off_f,
    )
    w.line(f"nisa.tensor_copy({out_expr}, psum_tile[0:{p_tile_M}, 0:{f_tile_N}])")


@_register_body("NKIActivationReduce", "reduce_close")
def _body_ar_reduce_close(w, op_graph, op, path_names, path_trips, forest, pipeline_dim=None, stage_offset=0) -> None:
    """Fold ``slot_vec`` into the op's ``(P, 1)`` output via ``nisa.tensor_reduce``.

    Runs after the F loop exits; the slot vector holds ``num_f_tiles``
    partial sums, one per F-tile. ``axis=2`` reduces the free axis of
    the 3D ``(p_tile, 1, num_f_tiles)`` slot_vec.
    """
    dst_name = op.output_names[0]
    dst = op_graph.tensors[dst_name]
    p_axis = op.axis_map["P"]
    p_tile = op_graph.dims[p_axis].tile_size
    f_axis = op.axis_map["F"]
    num_f = op_graph.dims[f_axis].num_tiles
    reduce_op = op.op_kwargs.get("reduce_op", "add")
    merge = _REDUCE_MERGE_OP[reduce_op]
    dst_p_slots = _tensor_total_slots(dst, dst.dim_ids[0], op_graph, forest)
    off_p = stage_offset if p_axis == pipeline_dim else 0
    p_slot = _slot_expr(path_names, path_trips, p_axis, dst_p_slots, off_p)
    dst_slot = f"{_sbuf_name(dst_name)}[0:{p_tile}, {p_slot}, 0:1]"
    slot_name = op.op_local_buffers["slot_vec"].emitted_name
    src_slot = f"{slot_name}[0:{p_tile}, 0:1, 0:{num_f}]"
    w.line(f"nisa.tensor_reduce({dst_slot}, {merge}, {src_slot}, axis=2)")


@_register_body("NKIActivationReduce", "reduce_step")
def _body_ar_reduce_step(w, op_graph, op, path_names, path_trips, forest, pipeline_dim=None, stage_offset=0) -> None:
    """Per-F-tile activation_reduce writing into a distinct slot of ``slot_vec``.

    The dst operand goes to the op-local scratch buffer (discarded);
    the reduce_res lands in ``slot_vec[0:p_tile, 0, f_slot:f_slot+1]``
    where ``f_slot`` is the current F-tile ordinal on the path. No
    prologue memset, no cross-tile merge — each call owns its slot.
    """
    src_name = op.operand_names["data"]
    src = op_graph.tensors[src_name]
    p_axis = op.axis_map["P"]
    f_axis = op.axis_map["F"]
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size
    num_f = op_graph.dims[f_axis].num_tiles
    act_op = op.op_kwargs.get("op", "copy")
    reduce_op = op.op_kwargs.get("reduce_op", "add")
    merge = _REDUCE_MERGE_OP[reduce_op]
    """slot_vec's F extent is num_f — the slot ordinal indexes directly into it."""
    off_f = stage_offset if f_axis == pipeline_dim else 0
    f_slot = _slot_expr(path_names, path_trips, f_axis, num_f, off_f)
    scratch_name = op.op_local_buffers["scratch"].emitted_name
    slot_name = op.op_local_buffers["slot_vec"].emitted_name
    scratch_slot = f"{scratch_name}[0:{p_tile}, 0, 0:{f_tile}]"
    reduce_res_slot = f"{slot_name}[0:{p_tile}, 0, {f_slot} : {f_slot} + 1]"
    src_p_slots = _tensor_total_slots(src, src.dim_ids[0], op_graph, forest)
    src_f_slots = _tensor_total_slots(src, src.dim_ids[1], op_graph, forest) if len(src.dim_ids) > 1 else 1
    src_off_p = stage_offset if src.dim_ids[0] == pipeline_dim else 0
    src_off_f = stage_offset if len(src.dim_ids) > 1 and src.dim_ids[1] == pipeline_dim else 0
    src_expr = _sbuf_tile_slice(
        _sbuf_name(src_name),
        src.dim_ids,
        p_tile,
        f_tile,
        path_names,
        path_trips,
        src_p_slots,
        src_f_slots,
        src_off_p,
        src_off_f,
    )
    w.line("nisa.activation_reduce(")
    w.indent()
    w.line(f"dst={scratch_slot},")
    w.line(f"op=nl.{act_op},")
    w.line(f"data={src_expr},")
    w.line(f"reduce_op={merge},")
    w.line(f"reduce_res={reduce_res_slot},")
    w.dedent()
    w.line(")")
